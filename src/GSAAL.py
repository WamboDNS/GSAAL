import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import tools.SubspaceSelection_2D as FB
from collections import defaultdict
from models.Generator import Generator
from models.Discriminator import Discriminator
from tensorflow import keras
import tensorflow as tf
tf.debugging.experimental.disable_dump_debug_info
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
import csv
from tqdm import tqdm

class GSAAL:
    def __init__(self, k = 20, batch_size = 500, stop_epochs = 30, lr_g = 0.001, lr_d = 0.01, seed = 777, momentum = 0.9, dataset="Arrhythmia"):
        self.storage = locals()
        self.train_history = defaultdict(list)
        self.k = k
        self.batch_size = batch_size
        self.stop_epochs = stop_epochs
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.momentum = momentum
        self.seed = seed
        self.dataset = dataset
        

    def fit(self, X, snap_path_res, snap_path_csv, X_test=None, Y_test=None):
        """Fit the GSAAL model to the data X.
        Args:
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        X_test : numpy array of shape (n_samples, n_features)
            The test samples, optional to plot AUC over epochs
        Y_test : numpy array of shape (n_samples, 1)
            The test labels, optional to plot AUC over epochs
        """
       
        self.__set_seed()
        epochs = self.stop_epochs + 10
        stop = 0 #Control variable for the training of the generator
        latent_size = X.shape[1]
        train_size = X.shape[0]
        
        
        ### ---------- Initialize GSAAL. Create one Generator, k Discriminators and correctly connect them ---------- ###
      
        generator = Generator(latent_size).model
        generator.compile(optimizer=keras.optimizers.SGD(learning_rate=self.lr_g, momentum=self.momentum, weight_decay = 0.1), loss='binary_crossentropy')
        
        latent_input = keras.Input(shape=(latent_size,))
        self.__set_seed()
        self.subspaces = FB.featureSubspaceSelection(latent_size, self.k)
        
        self.storage["sub_discriminator_sum"] = 0
        for i in range(self.k):
            self.storage["sub_discriminator" + str(i)] = Discriminator(len(self.subspaces[i]),train_size).model
            self.storage["fake_data" + str(i)] = generator(latent_input)
            self.storage["sub_discriminator" + str(i)].compile(optimizer=keras.optimizers.SGD(learning_rate=self.lr_d,momentum = self.momentum, weight_decay = 0.1), loss='binary_crossentropy')
            
            self.storage["sub_discriminator" + str(i)].trainable = False #Prevent training the discriminators while training the generator
            # In this step, each discriminator projects samples onto their own subspace
            self.storage["fake_data" + str(i)] = self.storage["sub_discriminator" + str(i)](tf.gather(self.storage["fake_data"+str(i)],self.subspaces[i],axis=1))
            self.storage["sub_discriminator_sum"] += self.storage["fake_data" + str(i)]
            self.storage["stop" + str(i)] = 0
            
            
        self.storage["sub_discriminator_sum"] /= self.k
        # Model with the mean decisions of Discriminators (mean Adversary). Used to train the generator.
        self.storage["combine_model"] = keras.Model(latent_input, self.storage["sub_discriminator_sum"])
        
        self.storage["combine_model"].compile(optimizer=keras.optimizers.SGD(learning_rate=self.lr_g, momentum = self.momentum, weight_decay = 0.1), loss='binary_crossentropy')
        
        ### ---------- End of GSAAL initialization ---------- ###
        
        step_snapshot = int(epochs / 5)
        
        ### ---------- START TRAINING LOOP ---------- ###
        for epoch in range(epochs):
            # create snapshot every 20% of training
            if epoch % step_snapshot:
                self.snapshot(snap_path_res + "/epoch_" + str(epoch), snap_path_csv)
            
            print('\nEpoch {} of {}'.format(epoch + 1, epochs))
            batch_size = min(self.batch_size, train_size)
            num_batches = int(train_size / batch_size)
            
            discriminator_loss = 0
            generator_loss = 0
            for id in range(self.k):
                self.storage["sub_discriminator_loss" + str(id)] = 0
                
            for i in range(num_batches):
                print('batch {}/{}:'.format(i + 1,num_batches))
                
                # Sample noise
                noise = np.random.uniform(0, 1, (int(batch_size), latent_size))
                
                data_batch = X[i * batch_size: (i + 1) * batch_size]
                self.storage["generated_data"] = generator.predict(noise, verbose = 0)
                
                batch = np.concatenate((data_batch, self.storage["generated_data"]))
                # 1 real data, 0 generated data
                batch_labels = np.array([1] * batch_size + [0] * int(batch_size))
                
                # Build Adversary loss. We stop the training of the subdisc. when around Nash equilibrium
                for id in tqdm(range(self.k), desc="Trained Discriminators: "):
                    if self.storage["stop" + str(id)] == 0: #If the discriminator is not in equilibrium, train it
                        self.storage["sub_discriminator_loss" + str(id)] += self.storage["sub_discriminator" + str(id)].train_on_batch(batch[:,self.subspaces[id]], batch_labels)
                    else:  #If the discriminator is in equilibrium, evaluate it
                        self.storage["sub_discriminator_loss" + str(id)] += self.storage["sub_discriminator" + str(id)].evaluate(batch[:,self.subspaces[id]], batch_labels, verbose = 0)
                    
                    if  self.storage["sub_discriminator_loss" + str(id)] <= 0.5 and epoch >= self.stop_epochs/self.k :
                        self.storage["stop" + str(id)] = 1
                    if  self.storage["sub_discriminator_loss" + str(id)] >= 1 and epoch >= self.stop_epochs/self.k :
                        self.storage["stop" + str(id)] = 0
                    

                # Build Generator loss
                if stop == 0:
                    trick = np.array([1] * batch_size)
                    generator_loss += self.storage["combine_model"].train_on_batch(noise, trick) / num_batches
                else:
                    print("\nEvaluating generator...")
                    trick = np.array([1] * batch_size)
                    generator_loss += self.storage["combine_model"].evaluate(noise, trick, verbose = 0) / num_batches
                
                if epoch + 1 > self.stop_epochs:
                    stop = 1
            
            if epoch % 10 == 0:
                self.snapshot(result_path="experiments/heatmaps/{}".format(self.dataset), 
                              csv_path="{}_story.csv".format(self.dataset), dataset=self.dataset, epoch=epoch, while_training=True) 
                
            self.train_history["generator_loss"].append(generator_loss)
            discriminator_loss = np.array([self.storage['sub_discriminator_loss' + str(id)]
                                           for id in range(self.k)]).sum()/(self.k*num_batches)
            self.train_history["discriminator_loss"].append(discriminator_loss)
            for id in range(self.k):
                self.train_history["sub_discriminator_loss" + str(id)].append(self.storage["sub_discriminator_loss" + str(id)] / num_batches)
            ### ---------- END TRAINING LOOP ---------- ###
            
            print("Number of discriminators stopped: " + str(np.array([self.storage["stop" + str(id)] for id in range(self.k)]).sum()) + " out of " + str(self.k))  
            
            #if np.array([self.storage["stop" + str(i)] for i in range(self.k)]).sum() == self.k:
            #    break
                

    def predict(self, X):
        """Return raw anomaly score of X.

        Args:
        X : numpy array of shape (n_samples, n_features) containing the data points to predict
        """
        
        p_value = self.storage["sub_discriminator" + str(0)].predict(X.to_numpy()[:,self.subspaces[0]])
        for id in range(1,self.k):
            p_value += self.storage["sub_discriminator" + str(id)].predict(X.to_numpy()[:,self.subspaces[id]])
        p_value /= self.k
        return p_value

    def get_params(self):
        return {'k': self.k, 'version': self.version, 'stop_epochs': self.stop_epochs, 'lr_g': self.lr_g, 'lr_d': self.lr_d, 'momentum': self.momentum, 'batch_size': self.batch_size, 'seed': self.seed}
    
    def update_seed(self, seed):
        self.seed = seed
    
    def __set_seed(self):
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.experimental.enable_op_determinism()
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        keras.utils.set_random_seed(self.seed) #seeds numpy, random and tf all at once
        os.environ["PYTHONHASSEED"] = str(self.seed)

    def __plot(self, result_path):
        """Hidden method for plotting the results of the training

        Args:

        result_path : string containing the Path to the result folder. If the folder does not exist, it will be created.
        """

        train_history = self.train_history
        k = self.k
        seed = self.seed
        plt.cla()
        plt.style.use('ggplot')
        dy = train_history['discriminator_loss']
        gy = train_history['generator_loss']
        for id in range(k):
            self.storage['dy_' + str(id)] = train_history['sub_discriminator_loss{}'.format(id)]
        x = np.linspace(1, len(gy), len(gy))
        fig, ax = plt.subplots()
        ax.plot(x, gy, color="cornflowerblue", label="Generator loss", linewidth=2)
        ax.plot(x, dy,color="crimson", label="Average discriminator loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        ax.legend(loc="upper right")
        
        fig_all,ax_all = plt.subplots()
        for i in range(k):
            ax_all.plot(x, self.storage['dy_' + str(i)], color="fuchsia", linewidth='0.5',alpha=0.3)
            
        ax_all.plot(x, gy, color="cornflowerblue", label="Generator loss", linewidth=2)
        ax_all.plot(x, dy,color="crimson", label="Average discriminator loss", linewidth=2)
        ax_all.legend(loc="upper right")
        plt.savefig(result_path + "/" " train_history " + str(seed) + ".pdf",format="pdf",dpi=1200)    
    

    def snapshot(self, result_path, csv_path, dataset,epoch, while_training=False):
        """Creates the snapshot of the model. It saves the plot, the parameters and the training history in the result_path folder as csv files and pdf images.

        Args:
        result_path : string containing the Path to the result folder. If the folder does not exist, it will be created.
        csv_path : string containing the name of the csv file where the parameters will be saved. If the file does not exist, it will be created.
        """
        if while_training==False:
            for i in range(self.k):
                self.storage["sub_discriminator" + str(i)].save(f"trained_models/{dataset}/disc_{i}.keras")
        else:
            for i in range(self.k):
                self.storage["sub_discriminator" + str(i)].save(f"trained_models/{dataset}/epoch{epoch}/disc_{i}.keras")
            return

        if not os.path.exists(result_path + "/sub_disc_stories/"):
            os.makedirs(result_path + "/sub_disc_stories/")
            
        output = pd.DataFrame()
        output["discriminator_loss"] = self.train_history["discriminator_loss"]
        output["generator_loss"] = self.train_history["generator_loss"]
        for i in range(self.k):
            output["sub_discriminator_loss" + str(i)] = self.train_history["sub_discriminator_loss" + str(i)]
        #Save all the story for auc and losses during training    
        output["discriminator_loss"].to_csv(result_path + "/disc_story" + str(self.seed) + ".csv",index=False)
        output["generator_loss"].to_csv(result_path + "/gen_story" + str(self.seed) + ".csv",index=False)
        
        for i in range(self.k):
            output["sub_discriminator_loss" + str(i)].to_csv(result_path + "/sub_disc_stories/" + "/sub_disc_story" +  "_" + str(i) + str(self.seed) + ".csv",index=False)
        self.__plot(result_path)
