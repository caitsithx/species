import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.autograd import Variable

import data_loader
import models
import report
import settings
from utils import save_weights, load_best_weights


def train_model(model, criterion, optimizer, lr_schedule, max_num=2,
                init_lr=0.001, num_epochs=100, data_loaders=None,
                fine_tune=False, pseudo=False):
    train_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    since = time.time()
    best_model = model
    best_acc = 0.0
    print(model.name)
    report.start(train_id, model.name)
    for epoch in range(num_epochs):
        epoch_since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                optimizer = lr_schedule(optimizer, epoch, init_lr=init_lr)
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0
            running_corrects = 0
            for data in tqdm.tqdm(data_loaders[phase]):
                inputs, labels, _ = data
                inputs, labels = Variable(inputs.cuda()), Variable(
                    labels.cuda())
                # print("input size: %s, %s" % (inputs.data.size()[0], inputs.data.size()[1]))
                optimizer.zero_grad()
                outputs = model(inputs)
                # preds = torch.sigmoid(outputs.data)
                # print("preds size:{}".format(preds.size()))
                # print("label size:{}".format(labels.data.size()))
                preds = torch.ge(outputs.data, 0.7).view(labels.data.size())
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds.int() == labels.data.int())
            print("running_loss %d" % running_loss)
            print("running_corrects %d" % running_corrects)
            print("data_num %d" % data_loaders[phase].num)
            epoch_loss = running_loss / data_loaders[phase].num
            epoch_acc = running_corrects / data_loaders[phase].num

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid':
                report.report_valid(epoch_loss, epoch_acc, running_corrects)

                if fine_tune:
                    weight_label = 'fine_tune'
                elif pseudo:
                    weight_label = 'pseudo'
                else:
                    weight_label = 'train'
                save_weights(epoch_acc, model, epoch, max_num=max_num,
                             weight_label=weight_label)
            else:
                report.report_train(epoch_loss, epoch_acc, running_corrects)
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model = copy.deepcopy(model)
                # torch.save(best_model.state_dict(), w_file)
        epoch_time = time.time() - epoch_since
        print('epoch {}: {:.0f}s'.format(epoch, epoch_time))
        report.report_time(epoch_time)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model


def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.6 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        print('existing lr = {}'.format(param_group['lr']))
        param_group['lr'] = lr
        report.report_lr(lr)
    return optimizer


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def cyc_lr_scheduler(optimizer, epoch, lr_decay_epoch=6):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    if epoch % lr_decay_epoch == 0 and epoch >= lr_decay_epoch:
        lr = lr * 0.6
    if lr < 5e-6:
        lr = 0.0001
    if epoch % lr_decay_epoch == 0 and epoch >= lr_decay_epoch:
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return optimizer


def train(model, fine_tune, pseudo, num_epochs=100, data_sets=None):
    init_lr = 0.0001
    criterion = nn.BCELoss()

    if fine_tune:
        arch = model.name

        if arch.startswith('resnet') or arch.startswith("inception"):
            dense_layers = model.fc
        elif arch.startswith("densenet") or arch.startswith("vgg"):
            dense_layers = model.classifier
        else:
            raise Exception('unknown model')

        optimizer_ft = optim.SGD(dense_layers.parameters(), lr=init_lr,
                                 momentum=0.9)
        init_lr = 0.001
    else:
        optimizer_ft = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)

    max_num = 2
    if pseudo:
        pseudo_data, valid_data = data_sets
        data_loaders = {
            'train': data_loader.get_pseudo_train_loader(model, pseudo_data),
            'valid': data_loader.get_val_loader(model, valid_data)}
        max_num += 2
    else:
        train_data, valid_data = data_sets
        data_loaders = {
            'train': data_loader.get_train_loader(model, train_data),
            'valid': data_loader.get_val_loader(model, valid_data)}

    model = train_model(model, criterion, optimizer_ft, lr_scheduler,
                        max_num=max_num, init_lr=init_lr,
                        num_epochs=num_epochs, data_loaders=data_loaders,
                        fine_tune=fine_tune, pseudo=pseudo)
    return model


def create_model(arch, fine_tune):
    print('Training {}...'.format(arch))
    model = models.create_model(arch, fine_tune=fine_tune)
    try:
        load_best_weights(model)
    except:
        print('Failed to load weights')
    if not hasattr(model, 'max_num'):
        model.max_num = 2
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs=2, help="train model")
parser.add_argument("--fine_tune", nargs=2, help="train model")
parser.add_argument("--pseudo", nargs=2, help="train model")

args = parser.parse_args()
if args.train:
    print('start training model')
    model_name = args.train[0]
    num_epochs = int(args.train[1])

    train_set = data_loader.get_train_set()
    valid_set = data_loader.get_valid_set()
    if model_name == '*':
        for name in settings.BATCH_SIZES:
            model = create_model(name, False)
            train(model, False, False, num_epochs=num_epochs,
                  data_sets=(train_set, valid_set))
            del model
    else:
        model = create_model(model_name, False)
        train(model, False, False, num_epochs=num_epochs,
              data_sets=(train_set, valid_set))
        del model
if args.fine_tune:
    print('start fine tune model')
    model_name = args.fine_tune[0]
    num_epochs = int(args.fine_tune[1])

    train_set = data_loader.get_train_set()
    valid_set = data_loader.get_valid_set()
    if model_name == '*':
        for name in settings.BATCH_SIZES:
            model = create_model(name, False)
            train(model, True, False, num_epochs=num_epochs,
                  data_sets=(train_set, valid_set))
            del model
    else:
        model = create_model(model_name, False)
        train(model, True, False, num_epochs=num_epochs,
              data_sets=(train_set, valid_set))
        del model
        # train_arch(model_name, True, False, None)
if args.pseudo:
    print('start training with pseudo labeling')
    model_name = args.pseudo[0]
    pseudo_label_file = args.pseudo[1]
    # train_arch(model_name, False, True, pseudo_label_file)

    print('done')
