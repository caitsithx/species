import argparse
import glob
import os

import time
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

import settings
import data_loader
from utils import create_model
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


def ensemble(tta=False):
    preds_raw = []

    w_files = glob.glob(settings.MODEL_DIR + os.sep + "*.pth")
    weights = {}
    for w_file in w_files:
        w_file = w_file.split(os.sep)[-1]
        acc = float(w_file.split('_')[-2])
        model_name = w_file.split('_')[0]
        if w_file.split('_')[1] == 'bn':
            model_name += '_bn'
        if w_file.split('_')[1] == 'v3':
            model_name += '_v3'

        if model_name in weights:
            weights[model_name].append(w_file)
            # existing_weight_file = weights[model_name]
            # existing_acc = float(existing_weight_file.split('_')[-2])
            # if acc > existing_acc:
            #     weights[model_name] = w_file
        else:
            weights[model_name] = [w_file]

    test_set = data_loader.get_test_set()
    for model_name, weight_files in weights.items():
        for weight_file in weight_files:
            full_w_file = settings.MODEL_DIR + os.sep + weight_file
            print("using saved weights: %s" % full_w_file)
            model = create_model(model_name, pre_trained=False)
            model.load_state_dict(torch.load(full_w_file))

            rounds = 1
            if tta:
                rounds = 10

            loader = data_loader.copy_test_loader(model, test_set, tta=True)

            for index in range(rounds):
                pred = np.array(make_preds(model, loader))
                preds_raw.append(pred)
            del model

    save_array(PRED_FILE_RAW, preds_raw)
    preds = np.mean(preds_raw, axis=0)
    save_array(PRED_FILE, preds)


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
parser.add_argument("--ens", action='store_true', nargs=1, help="ensemble predict")
parser.add_argument("--tta", action='store_true', nargs=1, help="ensemble predict")
parser.add_argument("--sub", action='store_true', help="generate submission file")

args = parser.parse_args()
if args.ens:
    ensemble()
    print('done')
if args.tta:
    ensemble(True)
    print('done')
if args.sub:
    print('generating submision file...')
    file_name = "submit-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + ".csv"
    submit(file_name)
    print('done')
    print('Please find submisson file at: {}'.format(settings.RESULT_DIR + os.sep + file_name))
