import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def handle_args_for_train(args):
    """
    Return the args as tuple
    :param args:
    :return:
    """
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    dim = args.dim
    optim = args.optimizer
    dp = args.dp
    return batch_size, epochs, lr, dim, optim, dp


def split(x, y, percent=0.1):
    """
    Split the given data-set with a given @percent
    :param x:
    :param y:
    :param percent:
    :return:
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(x, y, test_size=percent, random_state=42)


def generate_batch_indices(num_samples, batch_size):
    """
    Given batch size, and number of samples
    :param high:
    :param batch_size:
    :return:
    """
    import math
    sizes = [batch_size] * math.floor(num_samples / batch_size)  # truncating the last batch
    return [np.random.randint(low=0, high=num_samples, size=size) for size in sizes]


def plot_training_stats(d_loss, d_acc, g_loss):
    plt.plot(d_loss)
    plt.plot(g_loss, lw=0.5)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Discriminator Loss', 'Generator Loss'], loc='upper left')
    plt.show()
    plt.plot(d_acc)
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(['Discriminator Accuracy'], loc='upper left')
    plt.show()


def project_samples(samples, labels=None):
    """
    Project samples using t-SNE non-linear dimension reduction algorithm.
    If no labels given, assume all labels are 0.
    :param samples:
    :param labels:
    :return:
    """
    from sklearn.manifold import TSNE
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import numpy as np

    if tf.is_tensor(samples):
        samples = samples.numpy()

    if labels is None:
        test_predictions = np.zeros(samples.shape[0])  # assume generated samples.
    else:
        test_predictions = np.squeeze(labels)

    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(samples)
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = 2  # binary
    for lab in range(num_categories):
        indices = test_predictions == lab
        ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap(lab)).reshape(1, 4), label=lab,
                   alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()


def create_optimizer(lr):
    """
    Returns adam optimiser with a given lr
    :param lr:
    :return:
    """
    return tf.keras.optimizers.Adam(lr=lr, beta_1=0.5)


def project_real_data(args):
    x, y = pre_process_arff_data(args.dp)
    project_samples(x, y)


def pre_process_arff_data(filepath):
    """
    Given ARFF file, transforming nominal data into one-hot encoding
    and histack the result to the right.
    Note:
    - This function processing the whole data-set and assume it will fit into the memory.
    - Assumes the last column here represent the label.
    :param filepath: path to the arff file
    :return: numpy array represent the data, size: (num_samples, features), y as the labels vector.
    """
    from sklearn.preprocessing import minmax_scale
    from scipy.io import arff
    from sklearn import preprocessing
    data, meta = arff.loadarff(filepath)
    types = set(meta.types())
    print(f'Found types: {types}')
    x = None
    y = None
    cols = list(meta.names())
    for i, att in enumerate(cols):
        typo, classes = meta[att]

        if typo == 'nominal':
            res = preprocessing.label_binarize(data[att].astype(str), classes=classes)
        elif typo in ['real', 'numeric']:
            res = data[att]
            res = minmax_scale(res)  # min-max scale for all data
        else:
            res = data[att]

        if i == len(cols) - 1:  # assumes labels are last
            y = res
            continue

        if len(res.shape) == 1:
            res = np.expand_dims(res, axis=1)

        if x is None:
            x = res
        else:
            x = np.hstack([x, res])  # shape: (num_samples, x_right_shape + res_right_shape)

    return x, y


def plot_score_dist(y_pred):
    """
    Plot the score distribution of the prediction
    Note:
    Assumes binary labels.
    :param y_pred:
    :return:
    """
    plt.xlabel("Mean predicted value")
    plt.ylabel("Count")
    plt.title('Freq Dist of Positive class - Random Forest Classifier')
    plt.hist(y_pred[:, 1], range=(0, 1), bins=10, lw=2)
    plt.show()
    plt.xlabel("Mean predicted value")
    plt.ylabel("Count")
    plt.title('Freq Dist of Negative class - Random Forest Classifier')
    plt.hist(y_pred[:, 0], range=(0, 1), bins=10, lw=2)
    plt.show()
