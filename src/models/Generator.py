from keras import Sequential
from keras import layers
from tensorflow import keras


class Generator:
    
    #Initialize one Generator
    def __init__(self, latent_size):
        generator = Sequential()
        generator.add(layers.Dense(latent_size, input_dim=latent_size, activation="relu", kernel_initializer=keras.initializers.Identity(gain=1.0)))
        generator.add(layers.Dense(latent_size, input_dim=latent_size, activation="relu", kernel_initializer=keras.initializers.Identity(gain=1.0)))
        generator.add(layers.Dense(latent_size, input_dim=latent_size, activation="relu", kernel_initializer=keras.initializers.Identity(gain=1.0)))
        generator.add(layers.Dense(latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
        input_shape = (latent_size,)
        latent = keras.Input(input_shape)
        gen = generator(latent)
        self.model = keras.Model(latent, gen)
        

