import os
import numpy as np
import options_parser
import data_loader_linear as data_loader
import linear_model


def main(options):
    train_data_dir = options.train_data_dir
    train_files = os.listdir(train_data_dir)

    test_data_dir = options.test_data_dir
    test_files = os.listdir(test_data_dir)

    print("Loading data")
    data = data_loader.load_training_data(train_data_dir, train_files)
    images, labels, label_dict, imsize = data

    test_data = data_loader.load_test_data(test_data_dir, test_files, label_dict)
    test_images, test_labels = test_data
    test_weights = np.ones((len(test_labels), 1))
    num_classes = len(label_dict)
    test_info = test_images, test_labels, test_weights
    has_test_data = False
    if len(test_images) > 0:
        has_test_data = True

    for key in label_dict:
        ex_count = np.sum(labels[:, label_dict[key]] == 1.)
        print("Number of training examples of " + str(key) + ": " + str(ex_count))

    print("Feature Mapping: ", label_dict)
    print("Image Size: " + str(imsize))

    train_info, val_info = data_loader.merge_and_split_data(images, labels)
    train_data, train_labels, train_weights = train_info
    val_data, val_labels, val_weights = val_info

    print("Merged and Split Training Data")
    print("Training Data Size: ", len(train_data))
    print("Validation Data Size: ", len(val_data))

    if has_test_data:
        for key in label_dict:
            ex_count = np.sum(test_labels[:, label_dict[key]] == 1.)
            print("Number of test examples of " + str(key) + ": " + str(ex_count))
        print("Test Data Size: ", len(test_images))

    linear_model.construct_model(train_info, val_info, test_info, has_test_data)

options = options_parser.setup_data_options()
print(options)

if __name__ == "__main__":
    main(options)
