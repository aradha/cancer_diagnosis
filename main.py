import os
import h5py
import numpy as np
import tensorflow as tf
import time
import pickle
import options_parser
import resnet_50, resnet_specialized, resnet_patches
import vgg_encoder_model
import data_loader
import utils
import model_runner
import data_queues


def main(options):
    train_data_dir = options.train_data_dir
    train_files = os.listdir(train_data_dir)

    test_data_dir = options.test_data_dir
    test_files = os.listdir(test_data_dir)

    data = data_loader.load_training_data(train_data_dir, train_files)
    images, labels, label_dict, imsize = data

    test_data = data_loader.load_test_data(test_data_dir, test_files, label_dict)
    test_images, test_labels = test_data

    num_classes = len(label_dict)
    test_info = test_images, test_labels
    has_test_data = False
    if len(test_images) > 0:
        has_test_data = True

    for key in label_dict:
        ex_count = np.sum(labels[:, label_dict[key]] == 1.)
        print("Number of examples of " + str(key) + ": " + str(ex_count))

    print("Feature Mapping: ", label_dict)
    print("Image Size: " + str(imsize))

    train_info, val_info = data_loader.merge_and_split_data(images, labels)
    train_data, train_labels = train_info
    val_data, val_labels = val_info

    print("Merged and Split Training Data")
    print("Training Data Size: ", len(train_data))
    print("Validation Data Size: ", len(val_data))

    if has_test_data:
        for key in label_dict:
            ex_count = np.sum(test_labels[:, label_dict[key]] == 1.)
            print("Number of test examples of " + str(key) + ": " + str(ex_count))
        print("Test Data Size: ", len(test_images))

    data_queue = data_queues.QueueManager(train_data, train_labels)

    placeholders = utils.create_placeholders(imsize, num_classes)
    indata, answer, is_training, keep_prob, learning_rate = placeholders

    runnables = vgg_encoder_model.setup_model(indata,
                                              answer,
                                              imsize,
                                              is_training,
                                              keep_prob,
                                              num_classes,
                                              learning_rate)

    train_step, loss, predictions, accuracy, summaries, fc_2 = runnables
    train_summary, val_summary, test_summary = summaries
    init = tf.global_variables_initializer()
    print("Setup Model")

    log_dir = options.tb_log_dir
    features_dir = options.features_dir

    with tf.device('/gpu:0'):
        with tf.Session() as sess:

            train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
            val_writer = tf.summary.FileWriter(log_dir + "/val", sess.graph)
            test_writer = tf.summary.FileWriter(log_dir + "/test", sess.graph)
            sess.run(init)
            patience = 0
            max_patience = 1000  # wait for 30 steps of val loss not decreasing
            min_val_loss = float("inf")
            epoch = 0
            while patience < max_patience:
                epoch += 1
                print("EPOCH: ", str(epoch))
                start = time.time()
                train_statistics = model_runner.process_train_data(data_queue,
                                                                   placeholders,
                                                                   runnables,
                                                                   train_writer,
                                                                   num_classes,
                                                                   sess)

                avg_train_loss, train_acc = train_statistics[:2]
                train_confusion_matrix = train_statistics[2]
                train_misclassified = train_statistics[3]
                end = time.time()

                val_statistics = model_runner.process_data(val_info,
                                                           placeholders,
                                                           runnables,
                                                           val_writer,
                                                           num_classes,
                                                           False,
                                                           sess)

                avg_val_loss, val_acc = val_statistics[:2]
                val_confusion_matrix = val_statistics[2]
                val_misclassified = val_statistics[3]
                val_features = val_statistics[4]

                if has_test_data:
                    test_statistics = model_runner.process_data(test_info,
                                                                placeholders,
                                                                runnables,
                                                                test_writer,
                                                                num_classes,
                                                                True,
                                                                sess)

                    avg_test_loss, test_acc = test_statistics[:2]
                    test_confusion_matrix = test_statistics[2]
                    test_misclassified = test_statistics[3]
                    test_features = test_statistics[4]

                    print("train_loss: " + str(avg_train_loss) + " val_loss: " + str(avg_val_loss) +
                          " test_loss: "  + str(avg_test_loss))
                else:
                    print("train_loss: " + str(avg_train_loss) + " val_loss: " + str(avg_val_loss))

                print("train_acc: " + str(train_acc))
                print("val_acc: " + str(val_acc))
                if has_test_data:
                    print("test_acc: ", str(test_acc))
                print("Training Time: ", str(end - start))

                if avg_val_loss < min_val_loss:
                    min_val_loss = avg_val_loss
                    print("Training Confusion:\n", train_confusion_matrix)
                    print("Validation Confusion:\n", val_confusion_matrix)
                    if has_test_data:
                        print("Test Confusion:\n", test_confusion_matrix)

                        utils.store_misclassified(train_misclassified,
                                                  val_misclassified,
                                                  test_misclassified=test_misclassified)
                    else:
                        utils.store_misclassified(train_misclassified,
                                                  val_misclassified)

                    pickle_name = [k + "_{}".format(v) for k, v in label_dict.items()]
                    pickle_name = '_'.join(pickle_name)
                    pickle_name = pickle_name + ".p"
                    final_features = {'val': val_features}
                    if has_test_data:
                      final_features['test'] = test_features
                    with open(os.path.join(features_dir, pickle_name), 'wb') as outf:
                      pickle.dump(final_features, outf, protocol=pickle.HIGHEST_PROTOCOL)

                    patience = 0
                else:
                    patience += 1


options = options_parser.setup_data_options()
print(options)

if __name__ == "__main__":
    main(options)
