import subprocess

ARFF_SETS = ["Arrhythmia", "Annthyroid", "Cardiotocography", "InternetAds", "Ionosphere", "SpamBase", "Waveform"]

# The number of epochs below has been evaluated by hand, following the instructions of the official implementation of MO GAAL
DATASETS = [("Annthyroid",478),("Arrhythmia",375),("cardio",1000),("Cardiotocography",667)
            ,("CIFAR10_0",125),("FashionMNIST_0",200),("fault",1000),("InternetAds",134),("Ionosphere",800),
            ("landsat",223),("letter",500),("mnist",625),("musk",130),("optdigits",34),("satellite",125),
            ("satimage-2",1000),("SpamBase",100),("speech",84),("SVHN_0",125),("Waveform",292),("WPBC",400),
            ("AD_cable",300),("20news_0",100)]

if __name__ == "__main__":
    for (current_data,epochs) in DATASETS:
        output = "python3 mo_gaal.py --path {} --k 10 --stop_epochs {} --whether_stop 1 --lr_d 0.01 --lr_g 0.0001 --decay 1e-6 --momentum 0.9".format(current_data, epochs)
        print("Running " + output)
        cur = subprocess.Popen(output, shell=True)
        cur.wait()
        cur.kill()
        