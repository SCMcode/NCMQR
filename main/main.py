from datasets import load_yaz,load_bakery
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from main.NNs import SQRNet,MQRNet,PMQRNet,MonoNet,NCMQRNet,fit,pred

Q=np.array([.5,.6,.7,.8,.9,.95],dtype=np.float32)


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


if __name__=="__main__":
    X,y=load_bakery(include_date=True,one_hot_encoding=True,return_X_y=True)
    X_train,X_val,X_test,y_train,y_val,y_test=time_split(X,y)
    X_train, X_val, X_test=X_train.drop(columns="date"),X_val.drop(columns="date"),X_test.drop(columns="date")

    models=[SQRNet,MQRNet,PMQRNet,MonoNet,NCMQRNet]
    lams=[.1,1,10,100]
    reg = 1e-1
    results={}
    for cls in models:
        best=(np.inf,None,None)
        for lam in (lams if cls is PMQRNet else [10]):
            m=fit(cls,X_train,y_train,X_val,y_val,reg=reg,lam=lam)
            score=pinball(y_val,pred(m,X_val))
            if score<best[0]: best=(score,reg,lam)
        score,lam=best
        m=fit(cls,X_train,y_train,X_val,y_val,reg=reg,lam=lam)
        test=pinball(y_test,pred(m,X_test))
        results[cls.__name__]={"reg":reg,"lam":lam if cls is PMQRNet else None,"val":score,"test":test}
        print(cls.__name__,results[cls.__name__])