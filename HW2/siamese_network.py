from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
import tensorflow.keras.backend as K
from backbone_CNN3 import BackboneCNN
from backbone_CNN import BackboneCNN
from tensorflow.keras import Sequential
import numpy as np

class SiameseNetwork(keras.Model):
    def __init__(self, input_shape, augment=False):
        super(SiameseNetwork, self).__init__()
        self.augment = augment
        self.aug = Sequential([
            tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.25, 0.25),
                                                                         width_factor=(-0.25, 0.25)),
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.15, 0.15)),
            tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
            tf.keras.layers.experimental.preprocessing.RandomFlip(),
            tf.keras.layers.experimental.preprocessing.RandomContrast(factor=(0.2, 0.2))
        ], name="Augmentation_Layer")
        self.backbone = BackboneCNN(input_shape)
        self.l1_distance = Lambda(
            lambda lst: K.abs(lst[0] - lst[1])
        )
        self.output_layer = Dense(1, activation='sigmoid', name="output")

    def call(self, inputs, training=None, mask=None):
        # input shape: (m, 2, 250, 250, 3)
        x1 = inputs[:, 0, :, :, :]
        x2 = inputs[:, 1, :, :, :]
        x1_out = self.backbone(x1)
        x2_out = self.backbone(x2)
        dist = self.l1_distance([x1_out, x2_out])
        prediction = self.output_layer(dist)

        return prediction