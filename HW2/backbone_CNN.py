import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten


class BackboneCNN(keras.Model):
    def __init__(self, input_shape):
        super(BackboneCNN, self).__init__()
        # Given input size: W1 x H1 x D1
        # K - num of filters, F - filter size, S = stride, P = padding
        # Conv output: W2 = [(W1 - F + 2P)/S] + 1, H2 = [(H2 - F + 2P)/S] + 1, D2 = K
        # maxpool output: W2 = [(W1 - F)/S] + 1 , H2 = [(H2 - F)/S] + 1, D2 = D1
        self.layer1 = Conv2D(filters=64, input_shape=input_shape, kernel_size=[10, 10], strides=[1, 1],
                             kernel_regularizer=tf.keras.regularizers.l2(2e-3), activation='relu', name="Conv1")
        self.pool1 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2])
        # self.dropout1 = Dropout(rate=0.1)
        self.batchnorm1 = BatchNormalization()
        self.layer2 = Conv2D(filters=128, kernel_size=[7, 7], strides=[1, 1],
                             kernel_regularizer=tf.keras.regularizers.l2(2e-3), activation='relu', name="Conv2")
        self.pool2 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2])
        # self.dropout2 = Dropout(rate=0.15)
        self.batchnorm2 = BatchNormalization()
        self.layer3 = Conv2D(filters=128, kernel_size=[4, 4], strides=[1, 1],
                             kernel_regularizer=tf.keras.regularizers.l2(2e-3), activation='relu', name="Conv3")
        self.pool3 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2])
        # self.dropout3 = Dropout(rate=0.1)
        self.batchnorm3 = BatchNormalization()
        self.layer4 = Conv2D(filters=256, kernel_size=[4, 4], strides=[1, 1],
                             kernel_regularizer=tf.keras.regularizers.l2(2e-3), activation='relu', name="Conv4")
        self.pool4 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2])
        # self.dropout4 = Dropout(rate=0.1)
        self.batchnorm4 = BatchNormalization()
        # self.layer5 = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], activation='relu',
        #                      kernel_regularizer=tf.keras.regularizers.l2(2e-1), name="Conv5")
        # self.pool5 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2])
        self.flatten = Flatten()
        self.dense1 = Dense(4096, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(2e-2),
                            name='Dense1')

    def call(self, inputs, training=None, mask=None):
        # Given input: (m, 250, 250, 3) , m = batch_size
        x = self.layer1(inputs)  # (m, 240, 240, 3)
        x = self.pool1(x)  # (m, 119, 119, 3)
        # x = self.dropout1(x, training)
        x = self.batchnorm1(x, training)
        x = self.layer2(x)  # (m, 113, 113, 128)
        x = self.pool2(x)  # (m, 56, 56, 128)
        # x = self.dropout2(x, training)
        x = self.batchnorm2(x, training)
        x = self.layer3(x)  # (m, 50, 50, 128)
        x = self.pool3(x)  # (m, 24, 24, 128)
        # x = self.dropout3(x, training)
        x = self.batchnorm3(x, training)
        x = self.layer4(x)  # (m, 22, 22, 256)
        x = self.pool4(x)  # (m, 10, 10, 256)
        # x = self.dropout4(x)
        x = self.batchnorm4(x, training)
        # x = self.layer5(x)  # (m, 10, 10, 256)
        # x = self.pool5(x)  # (m, 4, 4, 256)
        x = self.flatten(x)  # (m, 4096)
        x = self.dense1(x)  # (m, 4096)
        return x

# TODO
# decide on convolutional network architecture and implement
# implement the call method
