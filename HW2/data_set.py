import tensorflow as tf
from img_augmentations import augment_ds
import numpy as np


def create_dataset(filename, debug=False, augment=False, split=None, BATCH_SIZE=32):
    filenames, labels = _generate_filenames_labels(filename)
    if split:
        filenames, labels = shufflez(filenames, labels)
        [train_x, train_labels_x], [valid_x, valid_labels_x] = splitter(filenames, labels, split=0.2)
        train_ds = create_dataset_as_DS(train_x, train_labels_x, debug=debug, augment=augment, BATCH_SIZE=BATCH_SIZE)
        valid_ds = create_dataset_as_DS(valid_x, valid_labels_x, debug=debug, augment=augment, BATCH_SIZE=BATCH_SIZE)
        return train_ds, valid_ds

    return create_dataset_as_DS(filenames, labels, debug=debug, augment=augment, BATCH_SIZE=BATCH_SIZE)


def shufflez(filenames, labels):
    """
    Pre shuffle (not efficient but work because the data-set is small)
    :param filenames:
    :param labels:
    :return:
    """
    assert len(filenames) == len(labels)
    num_examples = len(filenames)
    indices = np.array(list(range(num_examples)))
    np.random.shuffle(indices)
    indices_as_list = list(indices)
    return [filenames[i] for i in indices_as_list], [labels[i] for i in indices_as_list]


def splitter(filenames, labels, split=0.2):
    """
    Pre split of the given data-set (not efficient but work because the data-set is small)
    :param filenames:
    :param labels:
    :param split:
    :return:
    """
    assert len(filenames) == len(labels)
    split_index = int((1 - split) * len(filenames))
    return (filenames[:split_index], labels[:split_index]), (filenames[split_index:], labels[split_index:])


def create_dataset_as_DS(filenames, labels, augment=False, debug=False, BATCH_SIZE=32):
    """
    Create data-set as Tensorflow Dataset object.
    :param augment: if True then apply randomly affine transformations on input
    :param filenames: list of the filenames corresponds to the data-set
    :param labels: list of the labels corresponds to the data-set
    :param debug: if true will create smaller data-sets
    :param BATCH_SIZE: the batch-size
    :return: list of data-sets according to split
    """
    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    ds = ds.map(parser)  # after this the data-set becomes tensors and labels
    ds = ds.shuffle(buffer_size=len(filenames), reshuffle_each_iteration=True)
    if augment:
        ds = augment_ds(ds)

    if debug:
        ds = ds.shard(10, index=0)

    # def print_recover(x, y):
    #     tf.print(msg, x)
    #     return y

    # ds = ds.enumerate().map(print_recover)

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


def parser(filenames_tensor, label):
    """
    Load the images and labels and convert it to tensors
    :param filenames_tensor:
    :param label:
    :return:
    """
    f1, f2 = filenames_tensor[0], filenames_tensor[1]
    i1_string = tf.io.read_file(f1)
    i2_string = tf.io.read_file(f2)
    image1_decoded = tf.image.decode_jpeg(i1_string, channels=3)
    image1 = tf.cast(image1_decoded, tf.float32)
    image1 = tf.image.per_image_standardization(image1)
    image2_decoded = tf.image.decode_jpeg(i2_string, channels=3)
    image2 = tf.cast(image2_decoded, tf.float32)
    image2 = tf.image.per_image_standardization(image2)

    image1.set_shape((250, 250, 3))
    image2.set_shape((250, 250, 3))

    return tf.stack([image1, image2], axis=0), tf.cast(label, tf.int32)


def _generate_file_name(name, idx):
    return f'lfw2/lfw2/{name}/{name}_{idx.zfill(4)}.jpg'


def _generate_filenames_labels(filename):
    """
    Reader function for the given txt file
    :param filename:
    :return:
    """
    filenames, labels = [], []
    with open(filename, 'r') as f:
        n = int(f.readline())
        for i in range(n):
            name, n1, n2 = f.readline().split()
            x1 = _generate_file_name(name, n1)
            x2 = _generate_file_name(name, n2)
            filenames.append((x1, x2))
            labels.append(1)  # the same person

        for i in range(n):
            name1, n1, name2, n2 = f.readline().split()
            x1 = _generate_file_name(name1, n1)
            x2 = _generate_file_name(name2, n2)
            filenames.append((x1, x2))
            labels.append(0)  # not the same person

    return filenames, labels


def _parse_func(filenames_tensor, label):
    """
    Load the images and labels and convert it to tensors
    :param filenames_tensor:
    :param label:
    :return:
    """

    def my_py_func(filenames, label):
        numpies = filenames.numpy()
        # label_as_int = label.numpy()
        f1, f2 = numpies[0], numpies[1]
        i1_string = tf.io.read_file(f1)
        i2_string = tf.io.read_file(f2)
        image1_decoded = tf.image.decode_jpeg(i1_string, channels=3)
        image1 = tf.cast(image1_decoded, tf.float32)
        image2_decoded = tf.image.decode_jpeg(i2_string, channels=3)
        image2 = tf.cast(image2_decoded, tf.float32)
        # lbl = tf.reshape(label, [1])
        # lbl.set_shape([1])
        return (image1, image2), label

    return tf.py_function(my_py_func,
                          inp=(filenames_tensor, label),
                          Tout=(tf.float32, tf.int32))