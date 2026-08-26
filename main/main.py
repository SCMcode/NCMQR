import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import t,wilcoxon
from sklearn.isotonic import IsotonicRegression
from datasets import load_bakery,load_yaz,load_SID
from main.NNs import SQRNet,MQRNet,PMQRNet,MonoNet,NCMQRNet,fit,pred


def time_split(X,y,train_ratio=.6,val_ratio=.2):
    X=X.copy(); X["date"]=pd.to_datetime(X["date"]); dates=np.sort(X["date"].unique())
    d1,d2=dates[int(len(dates)*train_ratio)],dates[int(len(dates)*(train_ratio+val_ratio))]
    tr=X["date"]<d1; va=(X["date"]>=d1)&(X["date"]<d2); te=X["date"]>=d2
    return X.loc[tr].reset_index(drop=True),X.loc[va].reset_index(drop=True),X.loc[te].reset_index(drop=True),y.loc[tr].reset_index(drop=True),y.loc[va].reset_index(drop=True),y.loc[te].reset_index(drop=True)

def fit_all(Xtr,ytr,Xv,yv,Xte,reg=.1,lam=1,seed=0,only=None):
    tf.keras.utils.set_random_seed(seed)
    specs=[("SQR",SQRNet,0),("RSQR",SQRNet,reg),("MQR",MQRNet,0),("RMQR",MQRNet,reg),("PMQR",PMQRNet,0),("RPMQR",PMQRNet,reg),("Mono",MonoNet,0),("RMono",MonoNet,reg),("NCMQR",NCMQRNet,0),("RNCMQR",NCMQRNet,reg)]
    if only is not None:
        only=[only] if isinstance(only,str) else only
        specs=[s for s in specs if s[0] in only or (s[0]=="MQR" and any(x in only for x in ["Rearrangement","Isotonic"]))]

    P={}
    for name,cls,r in specs:
        tf.keras.backend.clear_session()
        P[name]=pred(fit(cls,Xtr,ytr,Xv,yv,reg=r,lam=lam),Xte)

    if "MQR" in P:
        p=P["MQR"]
        if only is None or "Rearrangement" in only:
            P["Rearrangement"]=np.sort(p,axis=1)
        if only is None or "Isotonic" in only:
            P["Isotonic"]=p.copy()
            idx=np.any(p[:,:-1]>p[:,1:]+1e-6,axis=1)
            iso=IsotonicRegression(increasing=True)
            P["Isotonic"][idx]=np.array([iso.fit_transform(Q,z) for z in p[idx]])

    if only is not None:
        P={k:v for k,v in P.items() if k in only}

    return P

def crossing(p,rtol=1e-6):
    gaps=p[:,:-1]-p[:,1:]
    tol=rtol*np.maximum(1,np.maximum(np.abs(p[:,:-1]),np.abs(p[:,1:])))
    gaps=np.where(gaps>tol,gaps,0.)
    crossed=gaps.sum(axis=1)>0
    return crossed.mean(),gaps[crossed].sum(axis=1).mean() if crossed.any() else 0.

def q_to_cu(q,co=1):
    return co*np.asarray(q)/(1-np.asarray(q))

def newsvendor_cost(y,p,cu,co=1):
    y=np.asarray(y).reshape(-1,1); cu=np.asarray(cu).reshape(1,-1)
    return np.where(p>=y,co*(p-y),cu*(y-p))

def mean_ci_str(x,digits=3):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if not n:return ""
    m=x.mean(); h=t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n) if n>1 else 0
    return f"{m:.{digits}f} (± {h:.{digits}f})"

def wilcox_p(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y); d=x[m]-y[m]
    if not len(d):return np.nan
    if np.allclose(d,0):return 1.
    try:return float(wilcoxon(d).pvalue)
    except ValueError:return np.nan

def improvement(base,new,p=np.nan,digits=2):
    b=np.mean(base); n=np.mean(new)
    if not np.isfinite(b) or abs(b)<1e-12:return ""
    s=f"{100*(b-n)/b:.{digits}f}%"
    return rf"$\mathbf{{{s}}}$" if np.isfinite(p) and p<.05 else s

def make_tables(df):
    models=df["Model"].unique()

    cross=pd.DataFrame({
        "Model":models,
        "CR":[mean_ci_str(df.loc[df.Model==m,"CR"]) for m in models],
        "CS":[mean_ci_str(df.loc[df.Model==m,"CS"]) for m in models]
    })

    rows=[]
    for m in models:
        row=[m]
        for q in Q:
            col=f"Cost_{q:g}"
            means=df.groupby("Model")[col].mean()
            best_model=means.idxmin()
            best=df.loc[df.Model==best_model].sort_values("Rep")[col].to_numpy()
            x=df.loc[df.Model==m].sort_values("Rep")[col].to_numpy()

            cost=mean_ci_str(x)
            gap=100*(x.mean()-best.mean())/best.mean()
            gap_str=f"{gap:.2f}%"

            if m==best_model:
                cost=rf"$\mathbf{{\underline{{{cost}}}}}$"
                gap_str=rf"$\mathbf{{\underline{{{gap_str}}}}}$"
            elif wilcox_p(x,best)>=.05:
                cost=rf"$\mathbf{{{cost}}}$"
                gap_str=rf"$\mathbf{{{gap_str}}}$"

            row.extend([cost,gap_str])
        rows.append(row)

    cols=["Model"]
    for q in Q:
        cols.extend([f"q={q:g} Cost",f"q={q:g} Gap (%)"])

    return cross,pd.DataFrame(rows,columns=cols)

def pinball(y,p):
    e=np.asarray(y).reshape(-1,1)-p
    return np.mean(np.maximum(Q*e,(Q-1)*e))

def run_replications(Xtr,ytr,Xv,yv,Xte,yte,n_rep=100,reg=.001,lam=1,only=None,out="replication_results.csv"):
    records=[]
    cu=q_to_cu(Q)

    for seed in range(n_rep):
        print(f"Replication {seed+1}/{n_rep}")

        try:
            P=fit_all(Xtr,ytr,Xv,yv,Xte,reg,lam,seed,only)

            for name,p in P.items():
                if not np.all(np.isfinite(p)):
                    raise ValueError(f"{name}: prediction contains NaN/Inf")

                cr,cs=crossing(p)
                costs=newsvendor_cost(yte,p,cu).mean(axis=0)

                if not np.all(np.isfinite(costs)):
                    raise ValueError(f"{name}: cost contains NaN/Inf")

                records.append([seed,name,cr,cs,*costs])

            pd.DataFrame(records,columns=["Rep","Model","CR","CS",*[f"Cost_{q:g}" for q in Q]]).to_csv(out,index=False)

        except Exception as e:
            print(f"FAILED seed={seed}: {type(e).__name__}: {e}")
            with open("failed_replications.txt","a") as f:
                f.write(f"seed={seed}: {type(e).__name__}: {e}\n")
            tf.keras.backend.clear_session()
            continue

    return pd.DataFrame(records,columns=["Rep","Model","CR","CS",*[f"Cost_{q:g}" for q in Q]])

if __name__=="__main__":
    Q=np.array([.5,.6,.7,.8,.9,.95],dtype=np.float32)
    ONLY = None#["MQR",'RMQR',"Rearrangement","Isotonic"]

    X,y=load_yaz(include_date=True,one_hot_encoding=True,return_X_y=True)
    print(X.shape,y.shape)
    Xtr,Xv,Xte,ytr,yv,yte=time_split(X,y)
    Xtr,Xv,Xte=[z.drop(columns="date").to_numpy(np.float32) for z in [Xtr,Xv,Xte]]
    ytr,yv,yte=[np.asarray(z,dtype=np.float32) for z in [ytr,yv,yte]]


    raw=run_replications(Xtr,ytr,Xv,yv,Xte,yte,n_rep=50,reg=1e-3,lam=1,only=ONLY)
    crossing_table,cost_table=make_tables(raw)

    print("\nCROSSING TABLE")
    print(crossing_table.to_string(index=False))

    print("\nCOST TABLE")
    print(cost_table.to_string(index=False))

    raw.to_csv("replication_results.csv",index=False)
    crossing_table.to_csv("crossing_table.csv",index=False)
    cost_table.to_csv("cost_table.csv",index=False)