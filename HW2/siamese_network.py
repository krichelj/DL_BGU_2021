from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
import tensorflow.keras.backend as K
from backbone_CNN import BackboneCNN


class SiameseNetwork(keras.Model):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = BackboneCNN()
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
