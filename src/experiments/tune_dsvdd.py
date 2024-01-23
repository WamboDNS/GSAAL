from get_dataset import load_dataset_path
import keras
import os
import pandas as pd
import scipy.io.arff as arff
import numpy as np
from datetime import date
import sklearn.metrics as metrics
import csv
from tqdm import tqdm
from pyod.models.deep_svdd import DeepSVDD


ARFF_SETS = ["Arrhythmia", "Annthyroid", "Cardiotocography", "InternetAds",
             "Ionosphere", "SpamBase", "Waveform"]

DATASETS = ["Annthyroid","Arrhythmia","cardio","Cardiotocography",
            "CIFAR10_0","FashionMNIST_0","fault","InternetAds",
            "Ionosphere","landsat","letter","mnist","musk","optdigits","satellite",
            "satimage-2","SpamBase","speech","SVHN_0","Waveform",
            "WPBC","Hepatitis","MVTec-AD_cable","20news_0"]

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
        path = "./datasets/elki/Arrhythmia_withoutdupl_norm_46.arff"
    elif dataset == "Annthyroid":
        path ="./datasets/elki/Annthyroid_withoutdupl_norm_07.arff"
    elif dataset == "Cardiotocography":
        path = "./datasets/elki/Cardiotocography_withoutdupl_norm_22.arff"
    elif dataset == "InternetAds":
        path = "./datasets/elki/InternetAds_withoutdupl_norm_19.arff"
    elif dataset == "Ionosphere":
        path = "./datasets/elki/Ionosphere_withoutdupl_norm.arff"
    elif dataset == "SpamBase":
        path = "./datasets/elki/SpamBase_withoutdupl_norm_40.arff"
    elif dataset == "Waveform":
        path = "./datasets/elki/Waveform_withoutdupl_norm_v01.arff"
    else:
        raise ValueError("Dataset not found")
    return load_data(path,"elki")

def load_data_wildcard(name):
    if name in ARFF_SETS:
        return get_elki_data(name), name.split("/")[-1].split(".")[0].split("_")[0]
    else:
        path = load_dataset_path(name)
        return load_data("datasets/" + path[0] + "/" + path[1]), name

def tune_dsvdd(train, test, labels, name,first):
    if first:
        with open("./tuning_results/deep_svdd_results/" +"highest.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset","epoch", "AUC"])
    ### grid set ###

    
    for n in tqdm([1], desc="DSVDD " + name):
        model = DeepSVDD()
        model.fit(train)
        scores = model.decision_function(test)
        fpr, tpr, thresholds = metrics.roc_curve(y_true = labels, y_score=scores)
        AUC = metrics.auc(fpr, tpr)
    
    with open("./tuning_results/deep_svdd_results/" +"default_dsvdd.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([name, str(AUC)])

### START ###
if __name__ == "__main__":
    first = False
    for current_data in DATASETS:
        print(current_data)
        (train, test, labels), name = load_data_wildcard(current_data)
        try:
            tune_dsvdd(train, test, labels, name,first)
        except:
            print("DSVDD failed")
            
        first = False
        