import tensorflow as tf


def create_dataset(filename, debug=False, split=None, BATCH_SIZE=32):
    filenames, labels = _generate_filenames_labels(filename)
    return create_dataset_as_DS(filenames, labels, debug=debug, split=split, BATCH_SIZE=BATCH_SIZE)


def create_dataset_as_DS(filenames, labels, debug=False, split=None, BATCH_SIZE=32):
    """
    Create data-set as Tensorflow Dataset object.
    :param filenames: list of the filenames corresponds to the data-set
    :param labels: list of the labels corresponds to the data-set
    :param debug: if true will create smaller data-sets
    :param split: split the data-set to 2 data-sets - value is between 0 to 1
    :param BATCH_SIZE: the batch-size
    :return: list of data-sets according to split
    """
    data_sets = []
    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    ds = ds.map(parser)
    ds = ds.shuffle(buffer_size=len(filenames), reshuffle_each_iteration=False)

    if debug:
        ds = ds.shard(10, index=0)

    if split is not None:
        split_every = int(1 / split)

        def is_valid(x, y):
            return x % split_every == 0

        def is_train(x, y):
            return not is_valid(x, y)

        recover = lambda x, y: y

        valid_dataset = ds.enumerate() \
            .filter(is_valid) \
            .map(recover)

        train_dataset = ds.enumerate() \
            .filter(is_train) \
            .map(recover)

        data_sets.extend([train_dataset, valid_dataset])

    else:
        data_sets.append(ds)

    data_sets = [ds.batch(BATCH_SIZE) for ds in data_sets]

    data_sets = [ds.prefetch(buffer_size=tf.data.AUTOTUNE) for ds in data_sets]

    return data_sets


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
