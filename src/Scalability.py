from MSS_GAAL import MSS_GAAL
import keras
import tensorflow as tf
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
os.environ["OMP_NUM_THREADS"] = "1"

def genenerate_data(n_features,n_datapoints):
    X_train = pd.DataFrame(np.random.uniform(0,1,(n_datapoints,n_features)))
    labels = pd.DataFrame(np.zeros(X_train.shape[0])) 
    labels.iloc[int(X_train.shape[0]/2):] = 1
    return X_train, labels

def parse_arguments(): 
    parser = argparse.ArgumentParser(description="Novelty detection experiments for MSS_GAAL")
    parser.add_argument("--gpu", type=int,default=0)
    parser.add_argument("--datapoint_list", type=str, default="100,300,500,1000,2500,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000")
    parser.add_argument("--feature_list", type=str, default="100")
    
    return parser.parse_args()

def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

if __name__ == '__main__':
    args = parse_arguments()
    
    print(tf.config.list_physical_devices('GPU'))
    time_taken = pd.DataFrame(columns=["n_features","n_datapoints","time_taken"])
    memory_taken = pd.DataFrame(columns=["n_features","n_datapoints","memory_taken"])
    args.feature_list = args.feature_list.split(",")
    args.datapoint_list = args.datapoint_list.split(",")
    for n_features in args.feature_list:
        for n_datapoints in args.datapoint_list:
            X_train, labels = genenerate_data(int(n_features),int(n_datapoints))   
            X_test, labels_test = genenerate_data(int(n_features),10)  # Small hack to avoid changing the code of GSAAL too much                                        
            k = 30
            model = MSS_GAAL(k=k, batch_size=500, stop_epochs=0)
            start_memory = get_process_memory()
            with tf.device("/device:GPU:" + str(args.gpu)):
                model.fit(X_train, X_test, labels_test)
            end_memory = get_process_memory()
            X_test, labels_test = genenerate_data(int(n_features),1000000) # Tensorflow loads the model first in memory during its first pass. This takes time
                                                                           # We take a large number of elements in inference to have a good error estimation at the end.   
            print(f"Starting inference evaluation with {n_datapoints} datapoints and {n_features} features")
            start = time.time()
            model.predict(X_test)
            end = time.time()
            print(f"Time taken for inference a single data object with {n_datapoints} datapoints and {n_features} features: {float(end-start)/X_test.shape[0]}")
            time_taken = time_taken._append({"n_features":n_features,"n_datapoints":n_datapoints,"time_taken":float(end-start)/X_test.shape[0]},ignore_index=True)
            memory_taken = memory_taken._append({"n_features":n_features,"n_datapoints":n_datapoints,"memory_taken":(end_memory-start_memory)},ignore_index=True)
    print(time_taken)
    print(memory_taken)
    day = datetime.date.today()
    time_taken.to_csv(f"experiments/Scalability/time_taken_{day}.csv")
                                          