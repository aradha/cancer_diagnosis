import numpy as np
import tensorflow as tf
import h5py


def permute_train_data(train_data, train_labels, train_weights):
    merged = list(zip(train_data, train_labels, train_weights))
    permuted = np.random.permutation(merged)
    train_data = [d[0] for d in permuted]
    train_labels = [d[1] for d in permuted]
    train_weights = [d[2] for d in permuted]
    return train_data, train_labels, train_weights


def histogram_equalization(image):
    L = 4096
    image = image.astype("uint16")
    r, c = image.shape
    image = np.where(image > L - 1, L - 1, image)

    buckets = np.bincount(image.flatten(), minlength=4096)
    buckets = buckets[1:] / (r * c * 1. - buckets[0])

    nonzero_cdf = np.cumsum(buckets)
    norm_cdf = (L - 1) * nonzero_cdf / nonzero_cdf[-1]
    cdf = np.zeros(len(norm_cdf) + 1)
    cdf[1:] = norm_cdf

    image = np.where(image >= 0, np.floor(cdf[image]), 0.0)

    return image


def shift_intensities(image, amount):
    if amount > 0:
        return np.where(image + amount > 4095, 4095, image + amount)
    else:
        return np.where(image + amount < 0, 0, image + amount)


def create_placeholders(imsize, num_classes):
    indata = tf.placeholder('float', shape=[None, imsize, imsize])
    answer = tf.placeholder('float', shape=[None, num_classes])
    is_training = tf.placeholder(tf.bool, shape=None)
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    #weights = tf.placeholder('float', shape=[None, 1])

    return (indata, answer, is_training, keep_prob, learning_rate)


def store_misclassified(train_misclassified, val_misclassified,
                        test_misclassified=None):
    with h5py.File("misclassified_training.hdf5", "w") as f:
        f.create_dataset('images', data=[pic for (pic, label) in train_misclassified])
        f.create_dataset('true_labels', data=[label for (pic, label) in train_misclassified])
    with h5py.File("misclassified_validation.hdf5", "w") as f:
        f.create_dataset('images', data=[pic for (pic, label) in val_misclassified])
        f.create_dataset('true_labels', data=[label for (pic, label) in val_misclassified])
    if test_misclassified is not None:
        with h5py.File("misclassified_test.hdf5", "w") as f:
            f.create_dataset('images', data=[pic for (pic, label) in test_misclassified])
            f.create_dataset('true_labels', data=[label for (pic, label) in test_misclassified])
