import numpy as np
import tensorflow as tf


def weight_variable(name, shape):
    initializer = tf.contrib.layers.variance_scaling_initializer()
    return tf.get_variable(name, shape=shape,
                           initializer=initializer)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(name, x, patch_size):
    with tf.name_scope(name):
        kernel_size = [1, patch_size, patch_size, 1]
        stride_size = [1, patch_size, patch_size, 1]
        pool = tf.nn.max_pool(x, ksize=kernel_size,
                              strides=stride_size, padding='SAME')
    return pool


def prelu(x, name):
    a = tf.Variable(0.25, name=name + "prelu")
    return tf.maximum(x, 0.0) + a * tf.minimum(x, 0.0)


def conv_module(name, inputs, kernel_size, num_input_channels,
                num_filters):

    with tf.name_scope(name):
        W_conv = weight_variable(name,
                                 [kernel_size, kernel_size,
                                  num_input_channels, num_filters])
        b_conv = bias_variable([num_filters])
        conv = prelu(conv2d(inputs, W_conv) + b_conv, name)

    return conv


def fc_module(name, inputs, input_size, output_size):

    W_fc = weight_variable(name, [input_size, output_size])
    b_fc = bias_variable([output_size])
    fc = tf.matmul(inputs, W_fc) + b_fc

    return fc


def setup_model(indata, target, imsize, is_training, keep_prob, num_classes,
                learning_r):

    image_size = imsize

    x_image = tf.reshape(indata, [-1, image_size, image_size, 1])

    # Image is initially 128 x 128
    kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    filter_list = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512]
    pool_sizes = [2, 2, 2, 2]  # Final image is 8 x 8
    num_hidden_units_1 = 4096
    num_hidden_units_2 = 4096
    num_output_units = num_classes

    norm_x = tf.contrib.layers.batch_norm(x_image,
                                          decay=0.9,
                                          epsilon=1e-5,
                                          scale=True,
                                          scope="First_batch_norm",
                                          updates_collections=None,
                                          is_training=is_training)

    conv_1_1 = conv_module("W_conv_1_1", norm_x, kernel_sizes[0],
                           1, filter_list[0])

    conv_1_2 = conv_module("W_conv_1_2", conv_1_1, kernel_sizes[1],
                           filter_list[0], filter_list[1])

    #pool_1 = max_pool("pool_1", conv_1_2, pool_sizes[0])
    #new_image_size = int(image_size / pool_sizes[0])
    ## Size 11
    new_image_size = image_size

    #conv_2_1 = conv_module("W_conv_2_1", pool_1, kernel_sizes[2],
    #                       filter_list[1], filter_list[2])
    conv_2_1 = conv_module("W_conv_2_1", conv_1_2, kernel_sizes[2],
                               filter_list[1], filter_list[2])
    conv_2_2 = conv_module("W_conv_2_2", conv_2_1, kernel_sizes[3],
                           filter_list[2], filter_list[3])

    #pool_2 = max_pool("pool_2", conv_2_2, pool_sizes[1])
    #new_image_size = int(new_image_size / pool_sizes[1])

    #conv_3_1 = conv_module("W_conv_3_1", pool_2, kernel_sizes[4],
    #                       filter_list[3], filter_list[4])
    conv_3_1 = conv_module("W_conv_3_1", conv_2_2, kernel_sizes[4],
                               filter_list[3], filter_list[4])
    conv_3_2 = conv_module("W_conv_3_2", conv_3_1, kernel_sizes[5],
                           filter_list[4], filter_list[5])
    conv_3_3 = conv_module("W_conv_3_3", conv_3_2, kernel_sizes[6],
                           filter_list[5], filter_list[6])

    pool_3 = max_pool("pool_3", conv_3_3, pool_sizes[2])
    new_image_size = int(new_image_size / pool_sizes[2])

    """
    conv_4_1 = conv_module("W_conv_4_1", pool_3, kernel_sizes[7],
                           filter_list[6], filter_list[7])
    conv_4_2 = conv_module("W_conv_4_2", conv_4_1, kernel_sizes[8],
                           filter_list[7], filter_list[8])
    conv_4_3 = conv_module("W_conv_4_3", conv_4_2, kernel_sizes[9],
                           filter_list[8], filter_list[9])

    pool_4 = max_pool("pool_4", conv_4_3, pool_sizes[3])

    new_image_size = int(new_image_size / pool_sizes[3])
    #"""
    #Important: only works for image sizes that are powers of 2
    #fc_input_dim = new_image_size * new_image_size * filter_list[-1]
    fc_input_dim = 6 * 6 * 256
    pool_flat = tf.reshape(pool_3,
                           [-1, fc_input_dim])

    fc_1 = tf.nn.relu(fc_module("W_fc_1", pool_flat, fc_input_dim, num_hidden_units_1))
    fc_1_drop = tf.nn.dropout(fc_1, keep_prob)
    fc_2 = tf.nn.relu(fc_module("W_fc_2", fc_1_drop, num_hidden_units_1, num_hidden_units_2))
    fc_2_drop = tf.nn.dropout(fc_2, keep_prob)

    logits = fc_module("W_fc_3", fc_2_drop, num_hidden_units_2, num_output_units)

    with tf.name_scope("loss"):
        softmax = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target)
        #weighted_softmax = tf.mul(weights, softmax)
        softmax_loss = tf.reduce_mean(softmax)

    train_loss = tf.summary.scalar('training_loss', softmax_loss)
    val_loss = tf.summary.scalar('val_loss', softmax_loss)
    test_loss = tf.summary.scalar('test_loss', softmax_loss)

    train_step = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(softmax_loss)


    predictions = tf.nn.softmax(logits)
    correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(target, 1))

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    train_acc = tf.summary.scalar('training accuracy', accuracy)
    val_acc = tf.summary.scalar('validation accuracy', accuracy)
    test_acc = tf.summary.scalar('test accuracy', accuracy)

    train_summary = tf.summary.merge([train_loss, train_acc])
    val_summary = tf.summary.merge([val_loss, val_acc])
    test_summary = tf.summary.merge([test_loss, test_acc])

    summaries = (train_summary, val_summary, test_summary)

    return (train_step, softmax_loss, predictions, accuracy, summaries, fc_2)
