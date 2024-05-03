import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyod.models.dif as dif 
import pyod.models.lunar  as lunar
import pyod.models.deep_svdd as deep_svdd
from pyod.models.abod import ABOD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.gmm import GMM
from pyod.models.ocsvm import OCSVM
from pyod.models.mo_gaal import MO_GAAL
import tensorflow as tf
import os
import matplotlib.ticker as mticker
import sklearn.metrics as metrics
import argparse
import ast

def parse_arguments(): 
    parser = argparse.ArgumentParser(description="OCC MV experiments for all methods")
    parser.add_argument("--methods", default=[LOF(),ABOD(),KNN(),IForest(),OCSVM(),GMM(),deep_svdd.DeepSVDD(),dif.DIF(),lunar.LUNAR()])
    parser.add_argument("--datasets", nargs='+', default=["banana","star","circle","L","spiral"], type=str)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--oponents", nargs='+', default=["sinusoidal_wave","X","noise"], type=str)

    return parser.parse_args()

if __name__ == '__main__':
    results_df = []
    args = parse_arguments()
    
    for j in range(args.iters):
        for name in args.datasets:
            X = pd.read_csv(f"experiments/synthetic_data/{name}.csv", header=None)
            X_full = pd.read_csv(f"experiments/synthetic_data/data_with_noise/{name}.csv", header=None)
            for vs in args.oponents:
                if vs == name: continue
                print(f"DATASET: {name} vs {vs}")
                if (vs == "noise"):
                    X_out = pd.read_csv(f"experiments/synthetic_data/data_with_noise/X.csv", header=None)
                    for i in range(0,60):
                        X_out[i] = np.random.uniform(size=np.shape(X_out)[0])  
                else: X_out = pd.read_csv(f"experiments/synthetic_data/data_with_noise/{vs}.csv", header=None)
                X_extra_in = X.sample(10000,replace=True)
                for i in range(2,60):
                    X_extra_in[i] = np.random.uniform(size=np.shape(X_extra_in)[0])
                X_test = pd.concat([X_out, X_extra_in]).reset_index(drop=True)
                y = np.concatenate([np.zeros(X_out.shape[0]), np.ones(X_extra_in.shape[0])])
                for method in args.methods:    
                    method.fit(X_full) 
                    scores = method.decision_function(X_test)
                    fpr, tpr, thresholds = metrics.roc_curve(y_true = y, y_score = scores)
                    results_df.append({"AUC":metrics.auc(fpr, tpr),"Method":"e" + method.__class__.__name__,"dataset":name,"vs":vs})
                    print({"AUC":metrics.auc(fpr, tpr),"Method":"e" + method.__class__.__name__,"dataset":name,"vs":vs})

                fegan = []
                with tf.device('/device:GPU:0'):
                    for i in range(14):
                        fegan.append(tf.keras.models.load_model(f"trained_models/{name}/disc_{i}.keras"))
                
                subspaces = pd.read_csv("subspaces.txt", delimiter=";", header=None)
                for i in range(len(subspaces[0])):
                    subspaces[0][i] = ast.literal_eval(subspaces[0][i])

                with tf.device('/device:GPU:0'):
                    scores = fegan[0].predict(X_test.iloc[:,subspaces[0][0]])
                    for i in range(1,14):
                        scores +=  fegan[i].predict(X_test.iloc[:, subspaces[0][i]]) #Probability of being an inlier 
                    scores /= 14
                fpr, tpr, thresholds = metrics.roc_curve(y_true = y, y_score = scores)
                results_df.append({"AUC":metrics.auc(fpr, tpr),"Method":"GSAAL","dataset":name,"vs":vs})
                print({"AUC":metrics.auc(fpr, tpr),"Method":"GSAAL","dataset":name,"vs":vs})

    results_df = pd.DataFrame(results_df)
    results_df.to_csv("experiments/results_df.csv") 


