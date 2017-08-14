import glob
import inspect
import os

import bcolz
import torch

import settings


def is_debugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


class WeightFile:
    def __init__(self, file_path):
        self.file_path = file_path
        w_file = file_path.split(os.sep)[-1]
        parts = w_file.split('-')
        self.model_name = parts[0]
        self.weight_label = parts[-2]
        self.accuracy = float(parts[-3])


def load_best_weights(model):
    w_files = glob.glob(os.path.join(settings.MODEL_DIR, model.name) + '*-pth')

    weights = {}
    weight_file = None
    for w_file in w_files:
        weight_file = WeightFile(w_file)

        if weight_file.model_name in weights:
            existing_weight_file = weights[weight_file.model_name]
            if weight_file.accuracy > existing_weight_file.accuracy:
                weights[weight_file.model_name] = weight_file
        else:
            weights[weight_file.model_name] = weight_file

    if weight_file:
        print('loading weight: {}'.format(weight_file.file_path))
        model.load_state_dict(torch.load(weight_file.file_path))


def save_weights(acc, model, epoch, max_num=2, weight_label=''):
    print("save_weights.")
    f_name = '{}-{}-{:.5f}-{}-pth'.format(model.name, epoch, acc, weight_label)
    w_file_path = os.path.join(settings.MODEL_DIR, f_name)

    w_files = glob.glob(os.path.join(settings.MODEL_DIR, model.name) + '*-pth')

    saved_weights = False
    same_label = False
    for w_file in w_files:
        weight_file = WeightFile(w_file)
        if weight_label == weight_file.weight_label:
            same_label = True
            if acc > weight_file.accuracy:
                saved_weights = True
                try:
                    os.remove(weight_file.file_path)
                except:
                    print('Failed to delete file: {}'.format(weight_file.file_path))
            break

    if saved_weights or not same_label:
        torch.save(model.state_dict(), w_file_path)
        print("saved weights: %s" % w_file_path)


def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def load_weights_file(model, w_file):
    model.load_state_dict(torch.load(w_file))
