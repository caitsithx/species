import argparse
import os
import time

import numpy as np
import pandas as pd
import torch.nn as nn
import tqdm
from torch.autograd import Variable

import data_loader
import settings
import models
import utils
from utils import save_array, load_array

PRED_FILE = settings.RESULT_DIR + os.sep + 'pred_ens.dat'
PRED_FILE_RAW = settings.RESULT_DIR + os.sep + 'pred_ens_raw.dat'
batch_size = 16

w_file_matcher = ['dense161*pth', 'dense201*pth', 'dense169*pth', 'dense121*pth', 'inceptionv3*pth',
                  'res50*pth', 'res101*pth', 'res152*pth', 'vgg16*pth', 'vgg19*pth']


def make_preds(net, loader):
    preds = []
    m = nn.Softmax()
    net.eval()
    for (img, _) in tqdm.tqdm(loader):
        inputs = Variable(img.cuda())
        outputs = net(inputs)
        pred = outputs.data.cpu().tolist()
        for p in pred:
            preds.append(p)
    return preds


def create_model(arch):
    print('Training {}...'.format(arch))
    model = models.create_model(arch, fine_tune=False, pre_trained=False)
    try:
        utils.load_best_weights(model)
    except:
        print('Failed to load weights')
    if not hasattr(model, 'max_num'):
        model.max_num = 2
    return model


def ensemble(model_name, file_name, tta=False):
    preds_raw = []

    model = create_model(model_name)

    test_set = data_loader.get_test_set()

    rounds = 1
    if tta:
        rounds = 20

    loader = data_loader.copy_test_loader(model, test_set, tta=True)

    for index in range(rounds):
        predictions = np.array(make_preds(model, loader))
        preds_raw.append(predictions)

    preds = np.mean(preds_raw, axis=0)

    save_array(settings.RESULT_DIR + os.sep + file_name, preds)


def submit(filename):
    # filenames = [f.split('/')[-1] for f, i in dsets.imgs]
    # filenames = get_stage1_test_loader('res50').filenames
    preds = load_array(PRED_FILE)
    print(preds[:100])
    subm_name = settings.RESULT_DIR + os.sep + filename
    df = pd.read_csv(settings.DATA_DIR + os.sep + 'sample_submission.csv')
    df['invasive'] = preds
    print(df.head())
    df.to_csv(subm_name, index=False)

    preds2 = (preds > 0.5).astype(np.int)
    df2 = pd.read_csv(settings.DATA_DIR + os.sep + 'sample_submission.csv')
    df2['invasive'] = preds2
    df2.to_csv(subm_name + '01', index=False)


parser = argparse.ArgumentParser()
parser.add_argument("--ens", action='store_true', help="ensemble predict")
parser.add_argument("--tta", nargs=1, help="tta predict")
parser.add_argument("--sub", action='store_true', help="generate submission file")

args = parser.parse_args()
if args.ens:
    ensemble()
    print('done')
if args.tta:
    model_name = args.tta[0]
    file_name = "predicts-" + model_name + "-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + ".dat"
    ensemble(model_name, file_name, True)
    print('done')
if args.sub:
    print('generating submision file...')
    file_name = "submit-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + ".csv"
    submit(file_name)
    print('done')
    print('Please find submisson file at: {}'.format(settings.RESULT_DIR + os.sep + file_name))
