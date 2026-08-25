import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import t,wilcoxon
from sklearn.isotonic import IsotonicRegression
from datasets import load_bakery
from main.NNs import SQRNet,MQRNet,PMQRNet,MonoNet,NCMQRNet,fit,pred

Q=np.array([.5,.6,.7,.8,.9,.95],dtype=np.float32)

def time_split(X,y,train_ratio=.6,val_ratio=.2):
    X=X.copy(); X["date"]=pd.to_datetime(X["date"]); dates=np.sort(X["date"].unique())
    d1,d2=dates[int(len(dates)*train_ratio)],dates[int(len(dates)*(train_ratio+val_ratio))]
    tr=X["date"]<d1; va=(X["date"]>=d1)&(X["date"]<d2); te=X["date"]>=d2
    return X.loc[tr].reset_index(drop=True),X.loc[va].reset_index(drop=True),X.loc[te].reset_index(drop=True),y.loc[tr].reset_index(drop=True),y.loc[va].reset_index(drop=True),y.loc[te].reset_index(drop=True)

def fit_all(Xtr,ytr,Xv,yv,Xte,reg=.1,lam=1,seed=0):
    tf.keras.utils.set_random_seed(seed)
    specs=[("SQR",SQRNet,0),("RSQR",SQRNet,reg),("MQR",MQRNet,0),("RMQR",MQRNet,reg),("PMQR",PMQRNet,0),("RPMQR",PMQRNet,reg),("Mono",MonoNet,0),("RMono",MonoNet,reg),("NCMQR",NCMQRNet,0),("RNCMQR",NCMQRNet,reg)]
    P={}
    for name,cls,r in specs:
        tf.keras.backend.clear_session()
        P[name]=pred(fit(cls,Xtr,ytr,Xv,yv,reg=r,lam=lam),Xte)
    p=P["MQR"]
    P["Rearrangement"]=np.sort(p,axis=1)
    iso=IsotonicRegression(increasing=True)
    P["Isotonic"]=np.array([iso.fit_transform(Q,z) for z in p])
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

def run_replications(Xtr,ytr,Xv,yv,Xte,yte,n_rep=100,reg=.1,lam=1):
    records=[]
    cu=q_to_cu(Q)
    for seed in range(n_rep):
        print(f"Replication {seed+1}/{n_rep}")
        P=fit_all(Xtr,ytr,Xv,yv,Xte,reg,lam,seed)
        for name,p in P.items():
            cr,cs=crossing(p)
            records.append([seed,name,cr,cs,newsvendor_cost(yte,p,cu).mean()])
    return pd.DataFrame(records,columns=["Rep","Model","CR","CS","Cost"])

def make_tables(df):
    models=df["Model"].unique()

    cross=pd.DataFrame({
        "Model":models,
        "CR":[mean_ci_str(df.loc[df.Model==m,"CR"]) for m in models],
        "CS":[mean_ci_str(df.loc[df.Model==m,"CS"]) for m in models]
    })

    means=df.groupby("Model")["Cost"].mean()
    best_model=means.idxmin()
    best=df.loc[df.Model==best_model].sort_values("Rep")["Cost"].to_numpy()
    rows=[]

    for m in models:
        x=df.loc[df.Model==m].sort_values("Rep")["Cost"].to_numpy()
        gap=100*(x.mean()-best.mean())/best.mean()
        cost=mean_ci_str(x)
        gap_str=f"{gap:.2f}%"

        if m==best_model:
            gap_str=rf"$\mathbf{{\underline{{{gap_str}}}}}$"
            cost=rf"$\mathbf{{\underline{{{cost}}}}}$"
        elif wilcox_p(x,best)>=.05:
            gap_str=rf"$\mathbf{{{gap_str}}}$"
            cost=rf"$\mathbf{{{cost}}}$"

        rows.append([m,cost,gap_str])

    cost=pd.DataFrame(rows,columns=["Model","Cost (95% CI)","Cost gap (%)"])
    return cross,cost,best_model


if __name__=="__main__":
    X,y=load_bakery(include_date=True,one_hot_encoding=True,return_X_y=True)
    Xtr,Xv,Xte,ytr,yv,yte=time_split(X,y)
    Xtr,Xv,Xte=Xtr.drop(columns="date"),Xv.drop(columns="date"),Xte.drop(columns="date")

    raw=run_replications(Xtr,ytr,Xv,yv,Xte,yte,n_rep=2,reg=1e-1,lam=1)

    crossing_table,cost_table,best_model=make_tables(raw)

    print(f"\nBest model: {best_model}")
    print("\nCROSSING TABLE")
    print(crossing_table.to_string(index=False))
    print("\nCOST TABLE")
    print(cost_table.to_string(index=False))
    raw.to_csv("replication_results.csv",index=False)
    crossing_table.to_csv("crossing_table.csv",index=False)
    cost_table.to_csv("cost_table.csv",index=False)