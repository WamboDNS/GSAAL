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
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.abod import ABOD
from pyod.models.gmm import GMM
from pyod.models.sod import SOD
from pyod.models.iforest import IForest


ARFF_SETS = ["Arrhythmia", "Annthyroid", "Cardiotocography", "InternetAds", 
             "Ionosphere", "SpamBase", "Waveform"]
RESULT_PATH = "./tuning_results/shallow_methods/"
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

### DEFINE MODELS ###
# To add a new model, simply follow the LOF template below. You only have to exchange the model and build your grid.

def tune_LOF(train, test, labels, name,first):
    n_neighbors = [5,8,10,13,15,18,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    
    lof_path = RESULT_PATH + "LOF/"
    if not os.path.exists(lof_path):
        os.makedirs(lof_path)
    with open(lof_path + name + ".csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow(["n_neigbors", "AUC"])
        
    if first:
        with open(lof_path +"highest_rest.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset","n_neigbors", "AUC"])
    ### grid set ###
    
    auc_max = 0
    n_max = 0
    
    for n in tqdm(n_neighbors, desc="LOF " + name):
        model = LOF(n_neighbors=n,novelty=True)
        model.fit(train)
        scores = model.decision_function(test)
        fpr, tpr, thresholds = metrics.roc_curve(y_true = labels, y_score=scores)
        AUC = metrics.auc(fpr, tpr)
        
        if AUC > auc_max:
            auc_max = AUC
            n_max = n
        
        with open(lof_path + name + ".csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow([str(n), str(AUC)])
    
    with open(lof_path+ "highest_rest.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([name, str(n_max), str(auc_max)])

def tune_KNN(train, test, labels, name,first):
    n_neighbors = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,25,30,35,40,45,50]
    
    knn_path = RESULT_PATH + "KNN/"
    if not os.path.exists(knn_path):
        os.makedirs(knn_path)
    with open(knn_path + name + ".csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow(["n_neigbors", "AUC"])
    if first:
        with open(knn_path +"highest_rest.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset","n_neigbors", "AUC"])
    ### grid set ###
    
    auc_max = 0
    n_max = 0
    
    for n in tqdm(n_neighbors, desc="KNN " + name):
        model = KNN(n_neighbors=n)
        model.fit(train)
        scores = model.decision_function(test)
        fpr, tpr, thresholds = metrics.roc_curve(y_true = labels, y_score=scores)
        AUC = metrics.auc(fpr, tpr)
        
        if AUC > auc_max:
            auc_max = AUC
            n_max = n
        
        with open(knn_path + name + ".csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow([str(n), str(AUC)])
    
    with open(knn_path +"highest_rest.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([name, str(n_max), str(auc_max)])
            
def tune_OCSVM(train, test, labels, name,first):
    nus = [0.01,0.1,0.2,0.4,0.6,0.8]
    gammas = [0.0001,0.001,0.01]
    
    ocsvm_path = RESULT_PATH + "OCSVM/"
    if not os.path.exists(ocsvm_path):
        os.makedirs(ocsvm_path)
    with open(ocsvm_path + name + ".csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow(["nu", "gamma", "AUC"])
    if first:
        with open(ocsvm_path +"highest_rest.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset","nu", "gamma", "AUC"])
            
    auc_max = 0
    nu_max = 0
    gamma_max = 0
    
    for nu in tqdm(nus, desc="OCSVM " + name):
        for gamma in gammas:
            model = OCSVM(nu=nu, gamma = gamma)
            model.fit(train)
            scores = model.decision_function(test)
            fpr, tpr, thresholds = metrics.roc_curve(y_true = labels, y_score=scores)
            AUC = metrics.auc(fpr, tpr)
            
            if AUC > auc_max:
                auc_max = AUC
                nu_max = nu
                gamma_max = gamma
            
            with open(ocsvm_path + name + ".csv", "a+") as f:
                writer = csv.writer(f)
                writer.writerow([str(nu), str(gamma), str(AUC)])
    with open(ocsvm_path +"highest_rest.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([name, str(nu_max), str(gamma_max), str(auc_max)])

def tune_ABOD(train, test, labels, name,first):
    n_neighbors = [2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,25,30,35,40,45,50]
    
    abod_path = RESULT_PATH + "ABOD/"
    if not os.path.exists(abod_path):
        os.makedirs(abod_path)
    with open(abod_path + name + ".csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow(["n_neigbors", "AUC"])
    if first:
        with open(abod_path +"highest_rest.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset","n_neigbors", "AUC"])
    ### grid set ###
    
    auc_max = 0
    n_max = 0
    
    for n in tqdm(n_neighbors, desc="ABOD " + name):
        model = ABOD(n_neighbors=n)
        model.fit(train)
        scores = model.decision_function(test)
        fpr, tpr, thresholds = metrics.roc_curve(y_true = labels, y_score=scores)
        AUC = metrics.auc(fpr, tpr)
        
        if AUC > auc_max:
            auc_max = AUC
            n_max = n
        
        with open(abod_path + name + ".csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow([str(n), str(AUC)])
    with open(abod_path +"highest_rest.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([name, str(n_max), str(auc_max)])

def tune_GMM(train, test, labels, name,first):
    components = [2,3,4,5,6,7,8,9,10,12,14,16,18,20]
    
    gmm_path = RESULT_PATH + "GMM/"
    if not os.path.exists(gmm_path):
        os.makedirs(gmm_path)
    with open(gmm_path + name + ".csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow(["components", "AUC"])
    if first:
        with open(gmm_path +"highest.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset","components", "AUC"])
    ### grid set ###
    
    auc_max = 0
    components_max = 0
    
    for n in tqdm(components, desc="GMM " + name):
        model = GMM(n_components=n)
        model.fit(train)
        scores = model.decision_function(test)
        fpr, tpr, thresholds = metrics.roc_curve(y_true = labels, y_score=scores)
        AUC = metrics.auc(fpr, tpr)
        
        if AUC > auc_max:
            auc_max = AUC
            components_max = n
        
        with open(gmm_path + name + ".csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow([str(n), str(AUC)])
    with open(gmm_path +"highest.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([name, str(components_max), str(auc_max)])

def tune_SOD(train, test, labels, name,first):
    # alpha = 0.8 is recommended in paper, refset < n_neighbors
    n_neighbors = [5,8,10,13,15,18,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    ref_sets = [3,5,8,10,15,20,30,50]
    
    sod_path = RESULT_PATH + "SOD/"
    if not os.path.exists(sod_path):
        os.makedirs(sod_path)
    with open(sod_path + name + ".csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow(["n_neigbors", "AUC"])
    if first:
        with open(sod_path +"highest_rest.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset","n_neigbors", "ref_set", "AUC"])
    ### grid set ###
    
    auc_max = 0
    n_max = 0
    ref_set_max = 0
    
    for n in tqdm(n_neighbors, desc="SOD " + name):
        for ref_set in ref_sets:
            if ref_set < n:
                model = SOD(n_neighbors=n, ref_set=ref_set)
                model.fit(train)
                scores = model.decision_function(test.to_numpy())
                fpr, tpr, thresholds = metrics.roc_curve(y_true = labels, y_score=scores)
                AUC = metrics.auc(fpr, tpr)
                
                if AUC > auc_max:
                    auc_max = AUC
                    n_max = n
                    ref_set_max = ref_set
                
                with open(sod_path + name + ".csv", "a+") as f:
                    writer = csv.writer(f)
                    writer.writerow([str(n), str(AUC)])
    with open(sod_path +"highest_rest.csv", "a+") as f: 
        writer = csv.writer(f)
        writer.writerow([name, str(n_max), str(ref_set_max), str(auc_max)])
        
def tune_iforest(train, test, labels, name,first):
    n_estimators = [50, 100, 200, 300]
    max_features = [0.4, 0.6, 0.8, 1.0]
    
    iforest_path = RESULT_PATH + "IFOREST/"
    if not os.path.exists(iforest_path):
        os.makedirs(iforest_path)
    with open(iforest_path + name + ".csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow(["n_restimators","max_features", "AUC"])
        
    if first:
        with open(iforest_path+"highest_rest.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["n_restimators","max_features", "AUC"])
    ### grid set ###
    
    auc_max = 0
    n_estimators_max = 0
    max_features_max = 0
    
    for n in tqdm(n_estimators, desc="IFOREST " + name):
        for max_feat in max_features:
            model = IForest(n_estimators=n,max_features=max_feat)
            model.fit(train)
            scores = model.decision_function(test)
            fpr, tpr, thresholds = metrics.roc_curve(y_true = labels, y_score=scores)
            AUC = metrics.auc(fpr, tpr)
            
            if AUC > auc_max:
                auc_max = AUC
                n_estimators_max = n
                max_features_max = max_feat
            
            with open(iforest_path + name + ".csv", "a+") as f:
                writer = csv.writer(f)
                writer.writerow([str(n), str(max_feat), str(AUC)])
    
    with open(iforest_path +"highest_rest.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([name, str(n_estimators_max), str(max_features_max), str(auc_max)])

### START ###
if __name__ == "__main__":
    first = True
    for current_data in DATASETS:
        # name used to automatically store results with params in correct files
        print(current_data)
        (train, test, labels), name = load_data_wildcard(current_data)

        ### start tuning ###
        try:
            tune_LOF(train, test, labels, name,first)
        except:
            print("LOF failed")
        
        try:
            tune_iforest(train, test, labels, name,first)
        except:
            print("IForest failed")
            
        try:
            tune_ABOD(train, test, labels, name,first)
        except:
            print("ABOD failed")
            
        try:
            tune_SOD(train, test, labels, name,first)
        except:
            print("SOD failed")
            
        try:
            tune_KNN(train, test, labels, name,first)
        except:
            print("KNN failed")
            
        try:
            tune_GMM(train, test, labels, name,first)
        except:
            print("GMM failed")
            
        first = False
        ### end tuning ###
        