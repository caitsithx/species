import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models

import settings

MODEL_DIR = settings.MODEL_DIR
model_urls = {
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


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

    model.batch_size = settings.FINE_TUNE_BATCH_SIZES[arch]

    model.name = arch

    return model
