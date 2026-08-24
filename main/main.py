from datasets import load_yaz,load_bakery
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd

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


if __name__=="__main__":
    X,y=load_bakery(include_date=True,one_hot_encoding=True,return_X_y=True)
    X_train,X_val,X_test,y_train,y_val,y_test=time_split(X,y)
    X_train, X_val, X_test=X_train.drop(columns="date"),X_val.drop(columns="date"),X_test.drop(columns="date")

