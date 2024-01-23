from GSAAL import GSAAL
import keras
import tensorflow as tf
import os
import pandas as pd
import scipy.io.arff as arff
import argparse
import numpy as np
from datetime import date
from pyod.models import lof
from pyod.models import iforest
import sklearn.metrics as metrics

def load_data():
    keras.utils.set_random_seed(777) #seeds numpy, random and tf all at once
    os.environ["PYTHONHASSEED"] = str(777) 
    
    train_x = pd.read_csv("synthetic_data/{}.csv".format(args.data), header=None)
    
    for i in range(2,60):
        train_x[i] = np.random.uniform(size=np.shape(train_x)[0])
    train_x.to_csv("synthetic_data/data_with_noise/{}.csv".format(args.data), header=False,index=False)
    
    return train_x

def parse_arguments(): 
    parser = argparse.ArgumentParser(description="Novelty detection experiments for GSAAL")
    parser.add_argument("--gpu", type=int,default=0)
    parser.add_argument("--data", default="banana")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--stop_epochs", type=int, default=50)
    parser.add_argument("--lr_g", type=float, default=0.001)
    parser.add_argument("--lr_d", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--momentum", type=float, default=0.9)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    X_train = load_data()

    k = 2*int(np.sqrt(X_train.shape[1]))
    model = GSAAL(k=k, batch_size=args.batch_size, stop_epochs=args.stop_epochs, lr_g=args.lr_g, lr_d=args.lr_d, seed=args.seed, momentum=args.momentum, dataset=args.data) #Edit: Added a variable to save the methods inside the same heatmap folder
    with tf.device("/device:GPU:" + str(args.gpu)):
        model.fit(X_train.sample(frac=1)) #Edit: Shuffle the data before training
    model.snapshot(result_path="experiments/heatmaps/{}".format(args.data), csv_path="{}_story.csv".format(args.data),dataset=args.data,epoch=args.stop_epochs + 1) #Edit: the variable needs to also come here, and the epoch is to know when the training ended