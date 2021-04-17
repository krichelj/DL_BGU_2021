import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D


class BackboneCNN(keras.Model):
    def __init__(self):
        super(BackboneCNN, self).__init__()
        self.layer1 = Conv2D
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

# TODO
# decide on convolutional network architecture and implement
# implement the call method