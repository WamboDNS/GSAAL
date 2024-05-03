import pyod.models.lof as lof
import pyod.models.knn as knn
import pyod.models.abod as abod
import os
import pandas as pd
import scipy.io.arff as arff
import argparse
import numpy as np
from datetime import date
import sklearn.metrics as metrics
import time
import datetime
import psutil
from sklearn.neighbors import LocalOutlierFactor as LOF
from joblib import parallel_backend

def genenerate_data(n_features,n_datapoints):
    X_train = pd.DataFrame(np.random.uniform(0,1,(n_datapoints,n_features)))
    labels = pd.DataFrame(np.zeros(X_train.shape[0])) 
    labels.iloc[int(X_train.shape[0]/2):] = 1
    return X_train, labels

def parse_arguments(): 
    parser = argparse.ArgumentParser(description="Novelty detection experiments for MSS_GAAL")
    parser.add_argument("--method", type=str,default="LOF")
    parser.add_argument("--benchmark_type", type=str, default="datapoint")
    
    return parser.parse_args()
    

if __name__ == "__main__":
    
    time_taken = pd.DataFrame(columns=["n_features","n_datapoints","time_taken"])
    memory_taken = pd.DataFrame(columns=["n_features","n_datapoints","memory_taken"])
    args = parse_arguments()
    if args.benchmark_type == "datapoint":
        feature_list = [100]
        datapoint_list = [100,300,500,1000,2500,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
    else:
        feature_list = [50,100,250,500,1000,5000]
        datapoint_list = [5000]
    for n_features in feature_list:
        for n_datapoints in datapoint_list:
            X_train, labels = genenerate_data(n_features,n_datapoints)    
            X_test, labels = genenerate_data(n_features,100000)                                         
            with parallel_backend('threading', n_jobs=1):
                if args.method == "LOF":
                    model = LOF(novelty=True)
                if args.method == "ABOD":
                    model = abod.ABOD()
                if args.method == "KNN":
                    model = knn.KNN()
                genenerate_data(n_features,n_datapoints)
                model.fit(X_train)
                
                start = time.time()
                model.predict(X_test)
                end = time.time()
                
            print(f"Time taken for inference a single data object with {n_datapoints} datapoints and {n_features} features: {float(end-start)/X_test.shape[0]}")
            time_taken = time_taken._append({"n_features":n_features,"n_datapoints":n_datapoints,"time_taken":float(end-start)/X_test.shape[0]},ignore_index=True)
    print(time_taken)
    print(memory_taken)
    day = datetime.date.today()
    time_taken.to_csv(f"experiments/Scalability/{args.method}_time_taken_{args.benchmark_type}_{day}.csv")