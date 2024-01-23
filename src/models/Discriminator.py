from keras import Sequential
from keras import layers
from tensorflow import keras
import numpy as np
class Discriminator:
    
    # Initialize one Discriminator for each subspace
    def __init__(self, subspace_size, n):
        discriminator = Sequential()
        discriminator.add(layers.Dense(np.ceil(np.sqrt(n)), input_dim=subspace_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
        discriminator.add(layers.Dense(np.ceil(np.sqrt(n)), input_dim=subspace_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
        discriminator.add(layers.Dense(np.ceil(np.sqrt(n)), input_dim=subspace_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
        discriminator.add(layers.Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
        input_shape=(subspace_size,)
        data = keras.Input(input_shape)
        discr = discriminator(data)
        self.model = keras.Model(data, discr)