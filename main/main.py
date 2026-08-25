from datasets import load_yaz,load_bakery
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from main.NNs import SQRNet,MQRNet,PMQRNet,MonoNet,NCMQRNet,fit,pred
from sklearn.isotonic import IsotonicRegression



def time_split(X,y,train_ratio=.6,val_ratio=.2):
    X=X.copy()
    X["date"]=pd.to_datetime(X["date"])
    dates=X["date"].sort_values().unique()
    d1=dates[int(len(dates)*train_ratio)]
    d2=dates[int(len(dates)*(train_ratio+val_ratio))]
    train=X["date"]<d1
    val=(X["date"]>=d1)&(X["date"]<d2)
    test=X["date"]>=d2
    return X.loc[train].reset_index(drop=True),X.loc[val].reset_index(drop=True),X.loc[test].reset_index(drop=True),y.loc[train].reset_index(drop=True),y.loc[val].reset_index(drop=True),y.loc[test].reset_index(drop=True)

def pinball(y,p):
    e=np.asarray(y).reshape(-1,1)-p
    return np.mean(np.maximum(Q*e,(Q-1)*e))

def fit_all(Xtr,ytr,Xv,yv,Xte,reg=1e-1,lam=1):
    preds={}
    models=[
        ("SQR",SQRNet,0),("RSQR",SQRNet,reg),
        ("MQR",MQRNet,0),("RMQR",MQRNet,reg),
        ("PMQR",PMQRNet,0),("RPMQR",PMQRNet,reg),
        ("Mono",MonoNet,0),("RMono",MonoNet,reg),
        ("NCMQR",NCMQRNet,0),("RNCMQR",NCMQRNet,reg)
    ]

    for name,cls,r in models:
        m=fit(cls,Xtr,ytr,Xv,yv,reg=r,lam=lam)
        preds[name]=pred(m,Xte)

    p=preds["MQR"]
    preds["Rearrangement"]=np.sort(p,axis=1)
    iso=IsotonicRegression(increasing=True)
    preds["Isotonic"]=np.array([iso.fit_transform(Q,z) for z in p])
    return preds

def crossing(p):
    gaps=np.maximum(p[:,:-1]-p[:,1:],0)
    crossed=gaps.sum(axis=1)>0
    cr=crossed.mean()
    cs=gaps[crossed].sum(axis=1).mean() if crossed.any() else 0
    return cr,cs

def q_to_cu(q,co=1):
    q=np.asarray(q)
    return co*q/(1-q)

def newsvendor_cost(y,p,cu,co=1):
    y=np.asarray(y).reshape(-1,1)
    cu=np.asarray(cu).reshape(1,-1)
    return np.where(p>=y,co*(p-y),cu*(y-p))

def evaluate_all(preds,y):
    cu=q_to_cu(Q)
    rows=[]
    for name,p in preds.items():
        cr,sev=crossing(p)
        cost=newsvendor_cost(y,p,cu)
        rows.append([name,cr,sev,cost.mean()])
    return pd.DataFrame(rows,columns=["Model","CrossRate","Severity","Cost"])


if __name__=="__main__":

    Q=np.array([.5,.6,.7,.8,.9,.95],dtype=np.float32)

    X,y=load_bakery(include_date=True,one_hot_encoding=True,return_X_y=True)
    X_train,X_val,X_test,y_train,y_val,y_test=time_split(X,y)
    X_train,X_val,X_test=X_train.drop(columns="date"),X_val.drop(columns="date"),X_test.drop(columns="date")
    preds=fit_all(X_train,y_train,X_val,y_val,X_test,reg=1e-1,lam=1)



    # reg=1e-1
    # lams=[.01,.1,1,10,100]
    # results={}
    # for lam in lams:
    #     m=fit(PMQRNet,X_train,y_train,X_val,y_val,reg=reg,lam=lam)
    #     val=pinball(y_val,pred(m,X_val))
    #     results[lam]=val
    #     print(f"lambda={lam:g}, val pinball={val:.4f}")
    # best_lam=min(results,key=results.get)
    # print(f"\nBest lambda: {best_lam:g}, val pinball: {results[best_lam]:.4f}")

    # '''
    # lambda=0.01, val pinball=34.2594
    # lambda=0.1, val pinball=34.3694
    # lambda=1, val pinball=34.1616
    # lambda=10, val pinball=34.3050
    # lambda=100, val pinball=34.2979
    # Best lambda: 1, val pinball: 34.1616
    # '''