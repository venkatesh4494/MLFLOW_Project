import pandas as pd
import os
import argparse
import numpy as np


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

def get_data():
    URL="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df=pd.read_csv(URL,sep=";")
        return df
    except Exception as e:
        raise e


def evaluate(actual,predicted):
    mbe=mean_absolute_error(actual,predicted)
    r2Score=r2_score(actual,predicted)
    rmse=np.sqrt(mean_squared_error(actual,predicted))
    return mbe,r2Score,rmse

def main(alpha,l1_ratio):

    data=get_data()

    train,test=train_test_split(data)

    train_x=train.drop(['quality'],axis=1)
    test_x=test.drop(['quality'],axis=1)

    train_y=train[['quality']]
    test_y =test[['quality']]


    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=123)
        lr.fit(train_x, train_y)
        pred = lr.predict(test_x)

        mbe, r2Score, rmse = evaluate(test_y, pred)

        print(f'Elastic net params: alpha:{alpha},l1_ratio:{l1_ratio}')
        print(f'Elastic net metrics: mbe:{mbe},rmse:{rmse},r2:{r2Score}')

        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)

        mlflow.log_metric("mbe",mbe)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2Score)

        mlflow.sklearn.log_model(lr, "model")

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--alpha","-a",type=float,default=0.5)
    args.add_argument("--l1_ratio","-l1",type=float,default=0.5)
    parsed_args=args.parse_args()

    try:
        main(alpha=parsed_args.alpha,l1_ratio=parsed_args.l1_ratio)
    except Exception as e:
        raise e