import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten, Activation, Add, GlobalAveragePooling2D
from tensorflow.python.keras.regularizers import l2


class ResNet18(keras.Model):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.entrance = Sequential([
            Conv2D(64, (7, 7), strides=(2, 2), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((3, 3), strides=(2, 2), padding='same')
        ])

        self.l1 = self.build_block(64)
        self.l2 = self.build_block(128, stride=2)
        self.l3 = self.build_block(256, stride=2)
        self.l4 = self.build_block(512, stride=2)

        self.pool = GlobalAveragePooling2D()

        self.flatten = Flatten()
        self.dense1 = Dense(4096, activation='sigmoid', name='Dense1')

    def build_block(self, filter_num, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicResidualBlock(filter_num, stride))
        res_blocks.add(BasicResidualBlock(filter_num, stride=1))
        return res_blocks

    def call(self, inputs, training=None, mask=None):
        # Given input: (m, 250, 250, 3) , m = batch_size
        x = self.entrance(inputs)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.pool(x)
        x = self.dense1(x)

        return x


class BasicResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicResidualBlock, self).__init__()

        self.conv1 = Conv2D(filters=filter_num, kernel_size=(3, 3), strides=(stride, stride), padding='same')
        self.bn1 = BatchNormalization()
        self.actv1 = Activation(tf.keras.activations.relu)

        self.conv2 = Conv2D(filters=filter_num, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn2 = BatchNormalization()
        self.actv2 = Activation(tf.keras.activations.relu)

        if stride != 1:
            self.residual = Conv2D(filter_num, (1, 1), strides=stride)
        else:
            self.residual = lambda x: x

        self.adder = Add()


    def call(self, inputs, training=None, mask=None):
        # first block
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.actv1(x)

        #second block
        x = self.conv2(x)
        x = self.bn2(x)

        x_skip = self.residual(inputs)

        x = self.adder([x, x_skip])
        x = self.actv2(x)

        return x


class IdentityResidualBlock(tf.keras.layers.Layer):
    def __init__(self, f1, f2):
        super(IdentityResidualBlock, self).__init__()

        self.conv1 = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))
        self.bn1 = BatchNormalization()
        self.actv1 = Activation(tf.keras.activations.relu)

        self.conv2 = Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))
        self.bn2 = BatchNormalization()
        self.actv2 = Activation(tf.keras.activations.relu)

        self.conv3 = Conv2D(filters=f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))
        self.bn3 = BatchNormalization()

        self.adder = Add()
        self.actv3 = Activation(tf.keras.activations.relu)

    def call(self, inputs, **kwargs):

        x_skip = inputs

        # first block
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.actv1(x)

        #second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.actv2(x)

        #third_block
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.adder([x, x_skip])
        x = self.actv3(x)

        return x


class ConvResidualBlock(tf.keras.layers.Layer):
    def __init__(self, f1, f2, s):
        super(ConvResidualBlock, self).__init__()

        self.conv1 = Conv2D(filters=f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))
        self.bn1 = BatchNormalization()
        self.actv1 = Activation(tf.keras.activations.relu)

        self.conv2 = Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))
        self.bn2 = BatchNormalization()
        self.actv2 = Activation(tf.keras.activations.relu)

        self.conv3 = Conv2D(filters=f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))
        self.bn3 = BatchNormalization()

        self.shortcut_conv = Conv2D(f2, kernel_size=(1,1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))
        self.bn_skip = BatchNormalization()

        self.adder = Add()
        self.actv3 = Activation(tf.keras.activations.relu)

    def call(self, inputs, **kwargs):

        x_skip = inputs

        # first block
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.actv1(x)

        #second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.actv2(x)

        #third_block
        x = self.conv3(x)
        x = self.bn3(x)

        #shortcut up
        x_skip = self.shortcut_conv(x_skip)
        x_skip = self.bn_skip(x_skip)

        x = self.adder([x, x_skip])
        x = self.actv3(x)

        return x