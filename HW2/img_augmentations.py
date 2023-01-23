import tensorflow as tf
import numpy as np


def augment_ds(ds: tf.data.Dataset):
    """
    Given Tensorflow Dataset s.t (input=(x1,x2), target=1/0), apply augmentation functions according to augs list
    :param ds: Tensorflow Dataset
    :param augs: list of functions to apply on images
    :return:
    """

    dataset = ds.map(_apply_random_transformations)

    dataset = dataset.map(lambda x: tf.clip_by_value(x, 0, 1))

    return dataset


def _translate(img):
    """
    Randomly translating the image at each axis
    :param img: shape : (1, ?, ?, ?)
    :return:
    """
    l = tf.keras.layers.experimental.preprocessing.RandomTranslation(
        height_factor=(-0.2, 0.2),
        width_factor=(-0.3, 0.3)
    )

    return l(img)

def _rotation(img):
    """
    Apply rotation with random theta
    :param img:
    :return:
    """
    theta = np.random.randint(low=-90, high=90)
    new_img = tf.keras.preprocessing.image.apply_affine_transform(
        x=img,
        theta=theta
    )

    return new_img


def _scaling(img):
    """
    Randomly scaling the image by random factor from 0 to 1 at each axis
    :param img: shape : (1, ?, ?, ?)
    :return:
    """
    zx = np.random.uniform()
    zy = np.random.uniform()

    new_img = tf.keras.preprocessing.image.apply_affine_transform(
        x=img,
        zx=zx,
        zy=zy
    )

    return new_img


def _shear(img):
    """
    Shear the image in random angle
    :param img:
    :return:
    """
    theta = np.random.randint(low=-45, high=45)
    new_img = tf.keras.preprocessing.image.apply_affine_transform(
        x=img,
        shear=theta
    )

    return new_img


def _apply_random_transformations(images: tf.Tensor, label):
    """
    Apply random transformations on the images tensor
    :param images: Tensor shape: (2,250,250,3)
    :return:
    """
    img1 = tf.expand_dims(images[0, :, :, :], axis=0)
    img2 = tf.expand_dims(images[1, :, :, :], axis=0)

    print(f"Welcome to augmentations!, given shape: {images.shape}")

    rands = np.random.random(size=(2, 4))  # 2 images and 4 possible transforms

    prob = 0.5

    if rands[0, 0] > prob:
        img1 = _translate(img1)

    if rands[1, 0] > prob:
        img2 = _translate(img2)

    return tf.concat([img1,img2], axis=0)