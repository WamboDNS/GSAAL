from GSAAL import GSAAL
from experiments.get_dataset import load_dataset_path
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

def load_data(path,type="adbench"):
    keras.utils.set_random_seed(777) #seeds numpy, random and tf all at once
    os.environ["PYTHONHASSEED"] = str(777) 
    
    if type != "adbench":
        arff_data = arff.loadarff(path)
        df = pd.DataFrame(arff_data[0])
    else:
       data = np.load(path)
       df = pd.DataFrame(data["X"])
       df["outlier"] = data["y"]
       df["id"] = df.index

    df["outlier"] = pd.factorize(df["outlier"], sort=True)[0] #Keep in mind: 0 inlier, 1 outlier
    
    inlier = df[df["outlier"] == 0]

    train = inlier.sample(frac=0.8)
    test = df.drop(train.index)
    
    train_x = train.drop(axis=1, labels=["outlier", "id"])
    test_x = test.drop(axis=1, labels=["outlier", "id"])
    test_y = pd.DataFrame(columns=["y"])
    test_y["y"] = test["outlier"]
    
    return train_x, test_x, test_y

def get_elki_data(dataset):
    if dataset == "Arrhythmia":
        path = "./experiments/datasets/elki/Arrhythmia_withoutdupl_norm_46.arff"
    elif dataset == "Annthyroid":
        path ="./experiments/datasets/elki/Annthyroid_withoutdupl_norm_07.arff"
    elif dataset == "Cardiotocography":
        path = "./experiments/datasets/elki/Cardiotocography_withoutdupl_norm_22.arff"
    elif dataset == "InternetAds":
        path = "./experiments/datasets/elki/InternetAds_withoutdupl_norm_19.arff"
    elif dataset == "Ionosphere":
        path = "./experiments/datasets/elki/Ionosphere_withoutdupl_norm.arff"
    elif dataset == "SpamBase":
        path = "./experiments/datasets/elki/SpamBase_withoutdupl_norm_40.arff"
    elif dataset == "Waveform":
        path = "./experiments/datasets/elki/Waveform_withoutdupl_norm_v01.arff"
    else:
        raise ValueError("Dataset not found")
    return load_data(path,"elki")

def buildPath(dataset):
    result_path = "../Results/Run_" + str(date.today()) + "_"+ dataset
    
    return result_path

def parse_arguments(): 
    parser = argparse.ArgumentParser(description="Novelty detection experiments for GSAAL")
    parser.add_argument("--gpu", type=int,default=0)
    parser.add_argument("--data", default="Arrhythmia")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--stop_epochs", type=int, default=50)
    parser.add_argument("--lr_g", type=float, default=0.001)
    parser.add_argument("--lr_d", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--momentum", type=float, default=0.9)

    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    if args.data in ["Arrhythmia", "Annthyroid","Cardiotocography","InternetAds","Ionosphere","SpamBase","Waveform"]:
        X_train, X_test, Y_test = get_elki_data(args.data)
    else:
        dataset = load_dataset_path(args.data)
        X_train, X_test, Y_test = load_data("datasets/" + dataset[0] + "/" + dataset[1])  

    sqrt = int(np.sqrt(X_train.shape[1]))
    log = int(np.log(X_train.shape[1]))
    k_list = [log, sqrt, 1.5*sqrt, 2*sqrt, 3*sqrt, 4*sqrt]
    for k in k_list:
        model = GSAAL(k=int(k), batch_size=args.batch_size, stop_epochs=args.stop_epochs, lr_g=args.lr_g, lr_d=args.lr_d,
                        seed=args.seed,momentum=args.momentum)
        with tf.device("/device:GPU:" + str(args.gpu)):
            model.fit(X_train, buildPath(args.data), "/" + args.data + ".csv", X_test, Y_test)
    model.snapshot(buildPath(args.data)+"_"+str(int(k)),"/" + args.data + ".csv")
