import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, ReLU, Layer

leaky_alpha = 0.4


class MiniBatchDiscrimination(Layer):
    def __init__(self, B, C, name='MiniBatchDiscrimination'):
        super().__init__(name=name)
        self.B = B  # number of added features
        self.C = C  # convolved kernel
        self.linear = tf.keras.layers.Dense(units=B * C)  # linear

    def __call__(self, input):
        x = self.linear(input)  # (batch_size, B * C)
        activation = tf.reshape(x, (-1, self.B, self.C))  # (batch_size, B, C)
        transposed = tf.transpose(activation, [1, 2, 0])  # (B, C, batch_size)
        # broadcasting the difference
        diffs = tf.expand_dims(activation, axis=3) - tf.expand_dims(transposed, axis=0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), axis=2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), axis=2)
        return tf.concat([input, minibatch_features], axis=1)


def create_discriminator(batch_size, input_shape, dim, minibatch_disc=False):
    input = Input(shape=input_shape, batch_size=batch_size)
    x = Dense(dim * 4, activation=LeakyReLU(alpha=leaky_alpha))(input)
    # x = Dropout(0.4)(x)
    x = Dense(dim * 3, activation=LeakyReLU(alpha=leaky_alpha))(x)
    # x = Dropout(0.4)(x)
    x = Dense(dim * 2, activation=LeakyReLU(alpha=leaky_alpha))(x)
    # x = Dropout(0.4)(x)
    x = Dense(dim, activation=LeakyReLU(alpha=leaky_alpha))(x)
    if minibatch_disc:
        x = MiniBatchDiscrimination(B=5, C=3)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input, outputs=x, name='discriminator')


def create_generator(batch_size, input_shape, dim, real_dim):
    input = Input(shape=input_shape, batch_size=batch_size)
    x = Dense(dim, activation=LeakyReLU(alpha=leaky_alpha))(input)
    x = Dense(dim * 2, activation=LeakyReLU(alpha=leaky_alpha))(x)
    x = Dense(dim * 4, activation=LeakyReLU(alpha=leaky_alpha))(x)
    x = Dense(real_dim, activation=ReLU(max_value=1))(x)
    return Model(inputs=input, outputs=x, name='generator')


def create_gan(discriminator, generator, data_shape):
    gan_input = Input(shape=(data_shape,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    return Model(inputs=gan_input, outputs=gan_output)