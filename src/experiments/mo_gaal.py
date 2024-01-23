from get_dataset import load_dataset_path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Input, Dense
from keras import Sequential, Model
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from tensorflow import keras
import math
import argparse
import scipy.io.arff as arff
import warnings
warnings.filterwarnings("ignore")
import csv
ARFF_SETS = ["Arrhythmia", "Annthyroid", "Cardiotocography", "InternetAds", "Ionosphere", "SpamBase", "Waveform"]

def parse_args():
    parser = argparse.ArgumentParser(description="Run MO-GAAL.")
    parser.add_argument('--path', nargs='?', default='Data/WDBC',
                        help='Input data path.')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of sub_generator.')
    parser.add_argument('--stop_epochs', type=int, default=300,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--whether_stop', type=int, default=1,
                        help='Whether or not to stop training generator after stop_epochs.')
    parser.add_argument('--lr_d', type=float, default=0.01,
                        help='Learning rate of discriminator.')
    parser.add_argument('--lr_g', type=float, default=0.0001,
                        help='Learning rate of generator.')
    parser.add_argument('--decay', type=float, default=1e-6,
                        help='Decay.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')
    return parser.parse_args()

# Generator
def create_generator(latent_size):
    gen = Sequential()
    gen.add(Dense(latent_size, input_dim=latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    gen.add(Dense(latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    latent = Input(shape=(latent_size,))
    fake_data = gen(latent)
    return Model(latent, fake_data)

# Discriminator
def create_discriminator():
    dis = Sequential()
    dis.add(Dense(math.ceil(math.sqrt(data_size)), input_dim=latent_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    dis.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    data = Input(shape=(latent_size,))
    fake = dis(data)
    return Model(data, fake)

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

# Plot loss history
def plot(train_history, name,dataset):
    dy = train_history['discriminator_loss']
    gy = train_history['generator_loss']
    auc_y = train_history['auc']
    for i in range(k):
        names['gy_' + str(i)] = train_history['sub_generator{}_loss'.format(i)]
    x = np.linspace(1, len(dy), len(dy))
    fig, ax = plt.subplots()
    ax.plot(x, dy, color='blue')
    ax.plot(x, auc_y, color='yellow', linewidth = '3')
    for i in range(k):
        ax.plot(x, names['gy_' + str(i)], color='green', linewidth='0.5')
    ax.plot(x, gy,color='red')
    plt.savefig("./tuning_results/mogaal_plots/tuned_plots/" + dataset + ".pdf",format="pdf",dpi=1200)

if __name__ == '__main__':
    train = True

    # initilize arguments
    args = parse_args()

    # initialize dataset
    (train_x, test_x, test_y), name = load_data_wildcard(args.path)
    data_size = train_x.shape[0]
    latent_size = train_x.shape[1]
    print("The dimension of the training data :{}*{}".format(data_size, latent_size))

    with tf.device("/device:GPU:0"):
        if train:
            train_history = defaultdict(list)
            names = locals()
            epochs = args.stop_epochs * 3
            stop = 0
            k = args.k

            # Create discriminator
            discriminator = create_discriminator()
            discriminator.compile(optimizer=SGD(learning_rate=args.lr_d, weight_decay=args.decay, momentum=args.momentum), loss='binary_crossentropy')

            # Create k combine models
            for i in range(k):
                names['sub_generator' + str(i)] = create_generator(latent_size)
                latent = Input(shape=(latent_size,))
                names['fake' + str(i)] = names['sub_generator' + str(i)](latent)
                discriminator.trainable = False
                names['fake' + str(i)] = discriminator(names['fake' + str(i)])
                names['combine_model' + str(i)] = Model(latent, names['fake' + str(i)])
                names['combine_model' + str(i)].compile(optimizer=SGD(learning_rate=args.lr_g, weight_decay=args.decay, momentum=args.momentum), loss='binary_crossentropy')

            # Start iteration
            for epoch in range(epochs):
                print('Epoch {} of {}'.format(epoch + 1, epochs))
                batch_size = min(500, data_size)
                num_batches = int(data_size / batch_size)

                for index in range(num_batches):
                    print('\nTesting for epoch {} index {}:'.format(epoch + 1, index + 1))

                    # Generate noise
                    noise_size = batch_size
                    noise = np.random.uniform(0, 1, (int(noise_size), latent_size))

                    # Get training data
                    data_batch = train_x[index * batch_size: (index + 1) * batch_size]

                    # Generate potential outliers
                    block = ((1 + k) * k) // 2
                    for i in range(k):
                        if i != (k-1):
                            noise_start = int((((k + (k - i + 1)) * i) / 2) * (noise_size // block))
                            noise_end = int((((k + (k - i)) * (i + 1)) / 2) * (noise_size // block))
                            names['noise' + str(i)] = noise[noise_start : noise_end ]
                            names['generated_data' + str(i)] = names['sub_generator' + str(i)].predict(names['noise' + str(i)], verbose=0)
                        else:
                            noise_start = int((((k + (k - i + 1)) * i) / 2) * (noise_size // block))
                            names['noise' + str(i)] = noise[noise_start : noise_size]
                            names['generated_data' + str(i)] = names['sub_generator' + str(i)].predict(names['noise' + str(i)], verbose=0)

                    # Concatenate real data to generated data
                    for i in range(k):
                        if i == 0:
                            X = np.concatenate((data_batch, names['generated_data' + str(i)]))
                        else:
                            X = np.concatenate((X, names['generated_data' + str(i)]))
                    Y = np.array([1] * batch_size + [0] * int(noise_size))

                    # Train discriminator
                    discriminator_loss = discriminator.train_on_batch(X, Y)
                    train_history['discriminator_loss'].append(discriminator_loss)

                    # Get the target value of sub-generator
                    p_value = discriminator.predict(train_x)
                    p_value = pd.DataFrame(p_value)
                    for i in range(k):
                        names['T' + str(i)] = p_value.quantile(i/k)
                        names['trick' + str(i)] = np.array([float(names['T' + str(i)])] * noise_size)

                    # Train generator
                    noise = np.random.uniform(0, 1, (int(noise_size), latent_size))
                    if stop == 0:
                        for i in range(k):
                            names['sub_generator' + str(i) + '_loss'] = names['combine_model' + str(i)].train_on_batch(noise, names['trick' + str(i)])
                            train_history['sub_generator{}_loss'.format(i)].append(names['sub_generator' + str(i) + '_loss'])
                    else:
                        for i in range(k):
                            names['sub_generator' + str(i) + '_loss'] = names['combine_model' + str(i)].evaluate(noise, names['trick' + str(i)])
                            train_history['sub_generator{}_loss'.format(i)].append(names['sub_generator' + str(i) + '_loss'])

                    generator_loss = 0
                    for i in range(k):
                        generator_loss = generator_loss + names['sub_generator' + str(i) + '_loss']
                    generator_loss = generator_loss / k
                    train_history['generator_loss'].append(generator_loss)

                    # Stop training generator
                    if epoch +1 > args.stop_epochs:
                        stop = args.whether_stop

                # Detection result
                p_value = discriminator.predict(test_x)
                p_value = pd.DataFrame(p_value)
                data_y = pd.DataFrame(test_y)
                result = np.concatenate((p_value,data_y), axis=1)
                result = pd.DataFrame(result, columns=['p', 'y'])
                result = result.sort_values('p', ascending=True)

                # Calculate the AUC
                inlier_parray = result.loc[lambda df: df.y == 0, 'p'].values
                outlier_parray = result.loc[lambda df: df.y == 1, 'p'].values
                sum = 0.0
                for o in outlier_parray:
                    for i in inlier_parray:
                        if o < i:
                            sum += 1.0
                        elif o == i:
                            sum += 0.5
                        else:
                            sum += 0
                AUC = '{:.4f}'.format(sum / (len(inlier_parray) * len(outlier_parray)))
                for i in range(num_batches):
                    train_history['auc'].append((sum / (len(inlier_parray) * len(outlier_parray))))
                print('AUC:{}'.format(AUC))

    plot(train_history, 'loss', name)
    with open("./tuning_results/mogaal_plots/" + name +".csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset","AUC"])
            writer.writerow([name, AUC])