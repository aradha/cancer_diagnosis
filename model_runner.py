import numpy as np
import tensorflow as tf

train_batch_count = 0
val_batch_count = 0
test_batch_count = 0
LEARNING_RATE = 5e-5
PREV_LOSS = float("inf")
PATIENCE = 0


def process_train_data(data_queue,
                       placeholders,
                       runnables,
                       writer,
                       num_classes,
                       sess):

    global train_batch_count, LEARNING_RATE, PREV_LOSS, PATIENCE
    indata, answer, is_training, keep_prob, learning_rate = placeholders
    train_step, loss, predictions, accuracy, summaries, fc_2 = runnables
    train_summary, _, _= summaries
    batch_size = 128
    total_loss = 0.0
    acc = 0.0
    confusion_matrix = np.zeros((num_classes, num_classes))
    misclassified_images = []
    num_batches = 0.0

    total_batches = 20

    for i in range(total_batches):
    #while not data_queue.all_data_seen():
        batch_data, batch_labels = data_queue.draw_fair_batch(batch_size)
        train_batch_count += 1
        num_batches += 1

        runnable_list = [train_step, loss, accuracy, predictions, train_summary, fc_2]
        dropout_prob = .5
        flag = True

        feeder = {indata:batch_data,
                  answer: batch_labels,
                  is_training: flag,
                  keep_prob: dropout_prob,
                  learning_rate: LEARNING_RATE}

        outputs = sess.run(runnable_list,
                           feed_dict=feeder)

        batch_loss, batch_acc, batch_pred, summary, _ = outputs[-5:]
        if batch_loss >= PREV_LOSS:
            PATIENCE += 1
        PREV_LOSS = batch_loss

        # Generate confusion matrices and collect misclassified images

        for idx in range(len(batch_pred)):
            c_idx = np.argmax(batch_pred[idx])
            r_idx = np.argmax(batch_labels[idx])
            confusion_matrix[r_idx, c_idx] += 1.
            if r_idx != c_idx:
                misclassified_images.append((batch_data[idx], r_idx))

        total_loss += batch_loss * len(batch_data)
        acc += batch_acc * len(batch_data)
        writer.add_summary(summary, train_batch_count)

    avg_loss = total_loss / (num_batches * batch_size)
    avg_acc = acc / (num_batches * batch_size)

    # Reset all flags when an epoch is complete
    data_queue.reset_seen_flags()

    return avg_loss, avg_acc, confusion_matrix, misclassified_images


def process_data(data_info,
                 placeholders,
                 runnables,
                 writer,
                 num_classes,
                 test_flag,
                 sess):

    global val_batch_count, test_batch_count, LEARNING_RATE

    data, labels = data_info
    indata, answer, is_training, keep_prob, learning_rate = placeholders
    train_step, loss, predictions, accuracy, summaries, fc_2 = runnables
    train_summary, val_summary, test_summary = summaries
    batch_size = 64
    batch_idx = 0
    total_loss = 0.0
    acc = 0.0
    confusion_matrix = np.zeros((num_classes, num_classes))
    misclassified_images = []
    feature_data = {}
    local_batch_count = 0

    while batch_idx < len(data):
        end = batch_idx + batch_size
        if end > len(data):
            end = len(data)
        if test_flag:
            test_batch_count += 1
            runnable_list = [loss, accuracy, predictions, test_summary, fc_2]
        else:
            val_batch_count += 1
            runnable_list = [loss, accuracy, predictions, val_summary, fc_2]

        feeder = {indata:data[batch_idx:end],
                  answer: labels[batch_idx:end],
                  is_training: False,
                  keep_prob: 1.,
                  learning_rate: LEARNING_RATE}

        outputs = sess.run(runnable_list,
                           feed_dict=feeder)

        batch_loss, batch_acc, batch_pred, summary, batch_feats = outputs[-5:]

        # Generate confusion matrices and collect misclassified images
        idx = batch_idx
        for pred in batch_pred:
            c_idx = np.argmax(pred)
            r_idx = np.argmax(labels[idx])
            confusion_matrix[r_idx, c_idx] += 1.
            if r_idx != c_idx:
                misclassified_images.append((data[idx], r_idx))
            idx += 1

        total_loss += batch_loss * (end - batch_idx)
        acc += batch_acc * (end - batch_idx)
        feature_data[local_batch_count] = {'data' : data[batch_idx:end],
                                            'correct labels' : labels[batch_idx:end],
                                            'predictions' : batch_pred,
                                            'features' : batch_feats}

        if test_flag:
            writer.add_summary(summary, test_batch_count)
        else:
            writer.add_summary(summary, val_batch_count)

        batch_idx += batch_size
        local_batch_count += 1

    avg_loss = total_loss / len(data)
    avg_acc = acc / len(data)
    return avg_loss, avg_acc, confusion_matrix, misclassified_images, feature_data
