import tensorflow as tf
from backbone_CNN import BackboneCNN
from siamese_network import SiameseNetwork
import numpy as np
import datetime

from data_set import create_dataset


def small_run_for_debug():
    input_shape = (16, 2, 250, 250, 3)
    x = np.random.randn(*input_shape)
    y = np.random.randint(0, 2, (16,))
    y = tf.one_hot(y, depth=2)

    logs_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)

    model = SiameseNetwork()

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.5), loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.run_eagerly = True

    model.build(input_shape)
    model.summary()

    model.fit(x, y, batch_size=2, epochs=6, validation_split=0.2, callbacks=[tensorboard_callback])


def train_siamese_debug(train_file_name, test_file_name, BATCH_SIZE=32):
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir = "logs/fit/" + time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)

    # save_dir = f"model_save/siamese_{time}.hdf5"
    # save_callback = tf.keras.callbacks.ModelCheckpoint(save_dir, monitor="loss", verbose=1, save_best_only=True,
    #                                                    mode="auto", period=1)

    train_ds, valid_ds = create_dataset(train_file_name, debug=True, split=0.2, BATCH_SIZE=BATCH_SIZE)
    test_ds = create_dataset(test_file_name, debug=True, BATCH_SIZE=BATCH_SIZE)

    model = SiameseNetwork((250, 250, 3))

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.5), loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.run_eagerly = True

    input_shape = (BATCH_SIZE, 2, 250, 250, 3)

    model.build(input_shape)

    model.summary()

    model.fit(train_ds,
              epochs=3,
              steps_per_epoch=None,
              validation_data=valid_ds,
              validation_steps=None,
              callbacks=[tensorboard_callback])

    model.evaluate(test_ds,
                   steps=None,
                   callbacks=[tensorboard_callback])


def train_siamese(train_file_name, test_file_name, BATCH_SIZE=32):
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir = "logs/fit/" + time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)

    save_dir = f"model_save/siamese_{time}.hdf5"
    save_callback = tf.keras.callbacks.ModelCheckpoint(save_dir, monitor="loss", verbose=1, save_best_only=True,
                                                       mode="auto", period=1)
    train_ds, valid_ds = create_dataset(train_file_name, augment=False, split=0.2, BATCH_SIZE=BATCH_SIZE)
    test_ds = create_dataset(test_file_name, BATCH_SIZE=BATCH_SIZE)

    model = SiameseNetwork((250, 250, 3), augment=True)

    initial_lr = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=55,
        decay_rate=0.99,
        staircase=True
    )

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.5),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    input_shape = (BATCH_SIZE, 2, 250, 250, 3)
    model.build(input_shape)

    model.fit(train_ds,
              epochs=15,
              steps_per_epoch=None,
              validation_data=valid_ds,
              validation_steps=None,
              callbacks=[tensorboard_callback, save_callback])

    model.evaluate(test_ds,
                   steps=None,
                   callbacks=[tensorboard_callback])


def load_model_eval(filepath, test_file_name, BATCH_SIZE=32):

    model = SiameseNetwork((250, 250, 3))
    input_shape = (BATCH_SIZE, 2, 250, 250, 3)
    model.build(input_shape)

    initial_lr = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=55,
        decay_rate=0.99,
        staircase=True
    )

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.5),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.load_weights(filepath)
    test_ds = create_dataset(test_file_name, BATCH_SIZE=BATCH_SIZE)

    print("Restored model, preparing to evaluate....")

    loss, acc = model.evaluate(test_ds,
                   steps=None,
                   verbose=2)

    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    print("Restored model, loss: {:5.2f}".format(loss))


if __name__ == "__main__":
    train_file_name = "pairsDevTrain.txt"
    test_file_name = "pairsDevTest.txt"
    train_siamese(train_file_name, test_file_name)

    # load_model_eval(filepath='./model_save/siamese_20210419-161845.hdf5', test_file_name=test_file_name)