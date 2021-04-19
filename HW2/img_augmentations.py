import tensorflow as tf
import numpy as np


def augment_ds(ds: tf.data.Dataset):
    """
    Given Tensorflow Dataset s.t (input=(x1,x2), target=1/0), apply augmentation functions according to augs list
    :param ds: Tensorflow Dataset
    :param augs: list of functions to apply on images
    :return:
    """

    dataset = ds.map(_apply_random_transformations, num_parallel_calls=4)

    dataset = dataset.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=4)

    return dataset


def _affine_transformation(img):
    """
    Apply affine transformation with random theta
    :param img:
    :return:
    """
    theta = tf.random.uniform(shape=[], minval=-45, maxval=45)
    new_img = tf.keras.preprocessing.image.apply_affine_transform(
        x=img,
        theta=theta
    )

    return new_img


def _translate(img):
    """
    Randomly translating the image
    :param img: shape : (1, ?, ?, ?)
    :return:
    """
    l = tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.2, 0.2),
                                                                     width_factor=(-0.2, 0.2))
    return l(img)


def _rotation(img):
    """
    Randomly rotating the image
    :param img: shape : (1, ?, ?, ?)
    :return:
    """
    # rotation between -0.15*2pi to 0.15*2pi =
    l = tf.keras.layers.experimental.preprocessing.RandomRotation((-0.15, 0.15))
    return l(img)


def _apply_random_transformations(images: tf.Tensor, label):
    """
    Apply random transformations on the images tensor
    :param images: Tensor shape: (2,250,250,3)
    :return:
    """
    img1 = images[0, :, :, :]
    img2 = images[1, :, :, :]

    rands = np.random.random(size=(2, 2))  # 2 images and 2 possible transforms

    if rands[0, 0] > 0.5:
        img1 = _affine_transformation(img1)
    if rands[1, 0] > 0.5:
        img1 = _rotation(img1)

    if rands[1, 0] > 0.5:
        img2 = _affine_transformation(img2)
    if rands[1, 1] > 0.5:
        img2 = _rotation(img2)

    return tf.stack([img1, img2], axis=0), label
