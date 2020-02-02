#!/usr/bin/env python3

import sys
import os
from config import Config
from engine import Engine
import argparse


def main():
    """
    Main entry point function. This is where it begins
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', default=None, help='json config file representing network, hyperparameters, etc')
    args = parser.parse_args()
    config_file = args.config_file
    assert config_file and os.path.exists(config_file), "%s doesn't exist?" % config_file
    config = Config(config_file)  # init config
    engine = Engine(config)  # init engine
    task = config.get_value('task', 'train')  # by default train always

    if task == 'train':
        engine.init_from_config()
        engine.train()
    else:
        # assume all other tasks are considered test for now (name ignored basically)
        engine.init_from_config(is_train=False)
        engine.test_model()

    sys.stdout.close()


if __name__ == '__main__':
    main()
