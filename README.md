# Generative Subspace Adversarial Active Learning for Unsupervised Outlier Detection

GSAAL focuses on detecting outliers while incorporating the information of the feature subspaces. This is the official implementation of GSAAL introduced by: [PAPER]

Please cite the paper if you use our code. Thank you :)

# Details

In a nutshell, GSAAL consists of one Generator and K Discriminators. Each of these K Discriminators trains over their own, unique Feature Subspace. The theoretical derivations in [Paper] show how this methodology can aid outlier detection in cases where outliers can not be detected in the full space of features. 

## The Role of Active Learning

GSAAL falls inside what are known as "Generative Adversarial Active Learning" (GAAL) methods. In a nutshell, these methods differentiate themselves from other GAN-based outlier detection methods by incorporating a secondary training step after the convergence of the GAN. In a nutshell, the process for a general GAAL method is as follows:

1. _GAN training_: A GAN with one generator $\mathcal{G}$ and one detector $\mathcal{D}$ is trained until an equilibrium of the regular $\min-\max$ game is reached.
2. _Active Learning_: $\mathcal{G}$ is fixed and no longer updated. The detector $\mathcal{D}$ continues to train with the same loss. The training stops whenever $\mathcal{D}$ reaches an optimum by itself.

[Zhu,2017] showed how step 2 can be explained as an active learning procedure. Our proposed methodology utilizes $k$ detectors fitted in random subspaces and 1 generator in the full space of features. We utilize the same GAAL 2-step procedure for the training of GSAAL, where step 2 is performed individually for all detectors. We refer to the original article for more information. 

# Environment

All necessary packages can be installed via:

```
pip install -r requirements.txt
```

# Quick Start
To launch the model in the same pipeline as done in the original article, run:

```
python Novelty.py
```
This code already handles the data and performs One-class classification experiments. If, in the contrary, one wishes to create it custom code for the execution, one can simply import the file `MSS_GAAL.py`. First, initialize the method by running
```
MSS_GAAL(k,batch_size,stop_epochs,lr_g,lr_d,seed, momentum)
```
Then, feed a pandas data frame containing the training data as in the following:
```
MSS_GAAL.fit(X_train,X_test,Y_test)
```
The parameters `X_test` and `Y_test` are for a test data set and the labels of it. They are optional, do not take a part in the training process of GSAAL (as is an unsupervised method), and are only there for visualization and testing purposes of the AUC evolution during training. 

The code in `Novelty.py` includes the possibility of changing the parameters for the training of the model, as detailed next.

# Parameters
GSAAL's training uses the following parameters:

```
--gpu               GPU to select (if CUDA detects multiple); default = 0
--data              Data to use from the benchmark in [Han et al., 2022] (see section Data); default = "Arrhythmia"
--k                 Number of Discriminators; default = 20
--batch_size        Batch size to train on: min(#samples, batch_size); default = 500
--stop_epochs       Training epochs for Generator; default = 30
--lr_g              Learning rate of Generator; default = 0.001
--lr_d              Learning rate of Discriminators; default = 0.01
--seed              Seed used for the data generation; default = 777
--momentum          Momentum of SGD; default = 0.9
```
`k` is the only training parameter constituting a hyperparameter for the method. We suggest the general use of $k = 30$, but have obtained pretty similar results with $k=20$ in preliminary experiments. Additionally, one could also select a varying parameter of $k = 2\sqrt{d}$, being $d$ the number of features of the training data. 

# Data
To run the experiments we have employed data from [Han et al., 2022]. If one wishes to run the same experiments we have run, see section **Experiments**. If one wishes to run the data from this benchmark, it suffices to run the script called `src/experiments/Download_datasets.py`. This script will download all original datasets from [Han et al., 2022] and save them in a folder called `datasets/`. It will also create a JSON file called `datasets_files_name.json`. This file is necessary for the code to run, and contains all dataset paths inside the downloaded directory. If one wishes to add a new dataset not included in [Han et al., 2022], it is enough to include it as a .npz file inside one of the subfolders inside `dataset`. After that, include its name in the `json` as the other datasets and add the corresponding path. **This is only for running the methods using Novelty.py or any of our experiment's code. If one wishes to run GSAAL in their custom code, refer to Quick Start**.


# Tuning of 'stop_epochs'
Our original article contains further information for the training of GSAAL. In essence, this has to be done in two steps, as specified in detail. The code in `MSS_GAAL.py` already handles this for you. The `stop_epoch` parameter the network initializes with is the number of iterations for step 1. After that, the network trains for $10$ extra epochs. We fixed this number as we found it to be enough in all scenarios we tested the method in during preliminary experiments.  How to select the number of `stop_epoch` is more critical. Our recommendation is to run GSAAL for a long number of epochs with default values (between 200 and 300). Observe whether the network converges into an optimum or not by utilizing the generated plots using `MSS_GAAL.snapshot()` (see method in `MSS_GAAL.py`). I.e: "If the graph of $\mathcal{G}$ has stabilized before the selected number of `stop_epochs`. 
If not, consider increasing the number of `stop_epochs` and retry. If nothing works, try changing the training parameters. In our One-class classification experiments, we managed to converge without changing the training parameters, only varying `stop_epochs`. 

_**Note:**__If the experiment has been run using Novelty.py, the graph automatically saved inside the folder `Results/Run_{date}_{dataset}`._


<img src='' width="400" height="300">


