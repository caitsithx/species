import os
from os import path

DATA_DIR = path.expanduser('~') + os.sep + 'dl_data' + os.sep + 'invasive-species-monitoring'
TRAIN_DIR = DATA_DIR + os.sep + 'train'
TEST_DIR = DATA_DIR + os.sep + 'test'
VALID_DIR = DATA_DIR + os.sep + 'valid'
RESULT_DIR = DATA_DIR + os.sep + 'result'

TRAIN_RESIZED_DIR = DATA_DIR + os.sep + 'train-640'
TEST_RESIZED_DIR = DATA_DIR + os.sep + 'test-640'
MODEL_DIR = DATA_DIR + os.sep + 'models'
BATCH_SIZE = 24

output_num=1

BATCH_SIZES = {
    "resnet50": 32,
    "resnet101": 16,
    "resnet152": 12,
    'densenet161': 19,
    'densenet169': 19,
    'densenet121': 19,
    'densenet201': 12,
    'vgg19_bn': 16,
    'vgg16_bn': 16,
    'inception_v3': 18,
    'inceptionresnetv2': 8
}

epochs = 100