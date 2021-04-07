import os
import argparse, errno
import tensorflow as tf
from tensorflow import keras

from runner import Runner

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--config_path', type=str, default='./config.json')
conf = parser.parse_args()


def check_env():
    # check deep learning platform
    print("TensorFlow: %s" %(tf.__version__))
    print("Keras: %s" %(keras.__version__))
 
    # check available gpu devices
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())


def check_dir():
    exp_dir = ['./experiments']
    def check(dirs):
        for i in range(len(dirs)):
            try:
                os.makedirs(dirs[i])
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
    check(exp_dir)


if __name__ =='__main__':
    
    args = ConfLoader(conf.config_path).opt

    print("Smart Farm Analytics v1.0")

    check_env()

    check_dir()
    
    runner = Runner(args)
    runner.run()