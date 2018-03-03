import argparse


def setup_data_options():
    options = argparse.ArgumentParser()

    options.add_argument('-t', action="store", dest='train_data_dir',
                         default='train_data')

    options.add_argument('-v', action="store", dest='test_data_dir',
                         default='test_data')

    options.add_argument('-l', action="store", dest='tb_log_dir',
                         default='tensorboard_logs')

    options.add_argument('-f', action="store", dest='features_dir', default='features')

    return options.parse_args()
