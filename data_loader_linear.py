import h5py
import numpy as np
import utils
from skimage.transform import rotate

# All files for training, validation must be stored in one directory
def load_training_data(dir_name, filenames):
    image_key = "images"
    label_key = "CellType"

    images = []

    sparse_labels = []
    label_idx = 0
    label_dict = {}

    imsize = 0.0

    for filename in filenames:
        fname = dir_name + "/" + filename
        with h5py.File(fname) as f:
            print(fname)
            image_set = f[image_key]
            print(len(image_set))
            print(image_set[0].shape)
            labels = [str(label) for label in f[label_key]]

            for idx in range(len(image_set)):
                image = image_set[idx]
                images.append(image)

                if labels[idx] in label_dict:
                    sparse_labels.append(label_dict[labels[idx]])
                else:
                    label_dict[labels[idx]] = label_idx
                    sparse_labels.append(label_idx)
                    label_idx += 1

                if imsize == 0.0:
                    imsize = image_set[idx].shape[0]

    dense_labels = np.zeros((len(sparse_labels), len(label_dict)))

    for i in range(len(sparse_labels)):
        dense_labels[i, sparse_labels[i]] = 1.

    return np.array(images), dense_labels, label_dict, imsize


def load_test_data(dir_name, filenames, label_dict):
    image_key = "images"
    label_key = "CellType"

    images = []

    sparse_labels = []
    imsize = 0.0

    for filename in filenames:
        fname = dir_name + "/" + filename
        with h5py.File(fname) as f:
            image_set = f[image_key]
            labels = [str(label)  for label in f[label_key]]

            for idx in range(len(image_set)):
                image = image_set[idx]
                # Must have same labels as train data
                if labels[idx] in label_dict:
                    sparse_labels.append(label_dict[labels[idx]])
                    images.append(utils.histogram_equalization(image))
                    #images.append(image)
    dense_labels = np.zeros((len(sparse_labels), len(label_dict)))

    for i in range(len(sparse_labels)):
        dense_labels[i, sparse_labels[i]] = 1.

    return np.array(images), dense_labels


def augment_dataset(data):
    new_data = []
    new_labels = []
    new_weights = []
    for image, label, weight in data:
        new_data.append(image)
        new_labels.append(label)
        new_weights.append(weight)

    return np.array(new_data), np.array(new_labels), np.array(new_weights)


def merge_and_split_data(all_data, all_labels):

    #permuted = np.random.permutation(list(zip(all_data, all_labels)))
    #all_data = np.array([p[0] for p in permuted])
    #all_labels = np.array([p[1] for p in permuted])

    class_idx = 0
    class_data = {}
    n, num_classes = all_labels.shape
    class_weights = {}
    for i in range(num_classes):
        num_examples = np.sum(all_labels[:, i] == 1.)
        class_weights[i] = (1. * n) / (num_examples)

    for idx in range(n):
        label = all_labels[idx].nonzero()[0][0]
        example = all_data[idx]
        class_weight = class_weights[label]
        if label in class_data:
            class_data[label].append((example, all_labels[idx], class_weight))
        else:
            class_data[label] = [(example, all_labels[idx], class_weight)]

    percent_split = .15  # 15 % of all data for validation
    split_idx = int(n * percent_split / num_classes)

    #"""
    # Can comment out this region for a different split method for validation set
    # By commenting out this region you can get an equal number of images of
    # each class in the validation set as opposed to an equal percentage.

    split_idx = -1 #int(n * percent_split / num_classes)
    for label in class_data:
        new_idx = int(len(class_data[label]) * percent_split)
        if split_idx != -1:
            split_idx = min(new_idx, split_idx)
        else:
            split_idx = new_idx
    #"""
    print("Split Idx: ", split_idx)

    train_merge = []
    val_merge = []

    for label in class_data:
        #val_merge += class_data[label][0:split_idx]
        #train_merge += class_data[label][split_idx:]
        val_merge += class_data[label][0:int(split_idx/2)]
        val_merge += class_data[label][-int(split_idx/2)::]
        total = len(class_data[label])
        train_merge += class_data[label][int(split_idx/2):total - int(split_idx/2)]

    val_merge = np.random.permutation(val_merge)

    train_merge = np.random.permutation(train_merge)

    train_data, train_labels, train_weights = augment_dataset(train_merge)

    # Can uncomment the code below if you don't want to augment the dataset
    #train_data = np.array([d[0] for d in train_merge])
    #train_labels = np.array([d[1] for d in train_merge])
    #train_weights = np.array([d[2] for d in train_merge])
    train_weights = train_weights.reshape(len(train_weights), 1)
    val_data = np.array([d[0] for d in val_merge])
    val_labels = np.array([d[1] for d in val_merge])
    val_weights = np.ones((len(val_labels), 1))
    #val_weights = np.array([d[2] for d in val_merge])
    #val_weights = val_weights.reshape(len(val_weights), 1)

    train = (train_data, train_labels, train_weights)
    val = (val_data, val_labels, val_weights)
    return train, val
