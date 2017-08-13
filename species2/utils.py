import glob
import os

import bcolz
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models

import settings
import inspect

MODEL_DIR = settings.MODEL_DIR

w_files_training = []

model_urls = {
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def is_debugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


class WeightFile:
    def __init__(self, file_path):
        self.file_path = file_path
        w_file = file_path.split(os.sep)[-1]
        parts = w_file.split('_')
        if len(parts) < 4:
            parts = w_file.split('-')
            self.model_name = parts[0]
        else:
            self.model_name = parts[0]
            if parts[1] == 'bn':
                self.model_name += '_bn'
            if parts[1] == 'v3':
                self.model_name += '_v3'

        self.accuracy = float(parts[-2])


def load_best_weights(model):
    w_files = glob.glob(os.path.join(MODEL_DIR, model.name) + '*.pth')
    max_acc = 0
    best_file = None
    for w_file in w_files:
        try:
            parts = w_file.split('_')
            if len(parts) < 4:
                parts = w_file.split('-')
            stracc = parts[-2]
            acc = float(stracc)
            if acc > max_acc:
                best_file = w_file
                max_acc = acc
            w_files_training.append((acc, w_file))
        except:
            continue
    if max_acc > 0:
        print('loading weight: {}'.format(best_file))
        model.load_state_dict(torch.load(best_file))


def save_weights(acc, model, epoch, max_num=2, weight_label=''):
    print("save_weights")
    f_name = '{}-{}-{:.5f}-{}.pth'.format(model.name, epoch, acc, weight_label)
    w_file_path = os.path.join(MODEL_DIR, f_name)
    if len(w_files_training) < max_num:
        w_files_training.append((acc, w_file_path))
        torch.save(model.state_dict(), w_file_path)
        return
    min_acc = 10.0
    index_min = -1
    for i, item in enumerate(w_files_training):
        val_acc, fp = item
        if min_acc > val_acc:
            index_min = i
            min_acc = val_acc
    print("current acc %s vs min acc %s" % (acc, min_acc))
    if acc > min_acc:
        torch.save(model.state_dict(), w_file_path)
        print("saved weights: %s" % w_file_path)
        try:
            os.remove(w_files_training[index_min][1])
        except:
            print('Failed to delete file: {}'.format(w_files_training[index_min][1]))
        w_files_training[index_min] = (acc, w_file_path)


def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def load_weights_file(model, w_file):
    model.load_state_dict(torch.load(w_file))


def create_model(arch, fine_tune=False, pre_trained=True):
    if pre_trained:
        print("=> using pre-trained model '{}'".format(arch))
        if arch.startswith("vgg") and arch.endswith("bn"):
            model = models.__dict__[arch]()
            model.load_state_dict(model_zoo.load_url(model_urls[arch]))
        else:
            model = models.__dict__[arch](pretrained=pre_trained)
    else:
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch]()

    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    if arch.startswith('resnet') or arch.startswith("inception"):
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, settings.output_num), nn.Sigmoid())
    elif arch.startswith("densenet"):
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Linear(num_ftrs, settings.output_num), nn.Sigmoid())
    elif arch.startswith('vgg'):
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, settings.output_num),
            nn.Sigmoid())

    if arch.startswith("inception_v3"):
        model.aux_logits = False

    model = model.cuda()

    model.batch_size = settings.BATCH_SIZES[arch]
    if fine_tune:
        if arch.startswith('vgg'):
            model.batch_size = 56
        else:
            model.batch_size = 128
    model.name = arch

    return model
