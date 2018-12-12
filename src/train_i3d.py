import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from . import videotransforms
from .charades_dataset import Charades as Dataset
from .pytorch_i3d import InceptionI3d


def log_init(log_file):
    with open(log_file, "w") as f:
        f.write("step,split,loc_loss,cls_loss,tot_loss\n")


def log_step(log_file, step, split, loc_loss, cls_loss, tot_loss):
    with open(log_file, "a") as f:
        f.write("{},{},{},{},{}\n".format(step, split, loc_loss, cls_loss,
                                          tot_loss))


def run(train_loader,
        val_loader,
        num_classes,
        init_lr=0.1,
        max_steps=64e3,
        mode='rgb',
        batch_size=2,
        save_model='',
        log_csv="log.csv"):

    dataloaders = {'train': train_loader, 'val': val_loader}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3, spatial_squeeze=True)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(num_classes)
    # i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(
        i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = 4  # accum gradient
    steps = 0
    log_init(log_csv)

    # train it
    while steps < max_steps:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs = data["video"]
                labels = data["label"]

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.upsample(
                    per_frame_logits, t, mode='linear')

                loss_func = nn.BCEWithLogitsLoss()
                loc_loss = loss_func(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data[0]

                cls_loss = loss_func(
                    torch.max(per_frame_logits, dim=2)[0],
                    torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data[0]

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data[0]
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    log_step(log_csv, steps, "train", loc_loss.data[0],
                             cls_loss.data[0],
                             loss.data[0] * num_steps_per_update)
                    if steps % 10 == 0:
                        print(
                            '{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'
                            .format(phase,
                                    tot_loc_loss / (10 * num_steps_per_update),
                                    tot_cls_loss / (10 * num_steps_per_update),
                                    tot_loss / 10))
                        # save model
                        torch.save(i3d.module.state_dict(),
                                   save_model + str(steps).zfill(6) + '.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                log_step(log_csv, steps, "val", tot_loc_loss / num_iter,
                         tot_cls_loss / num_iter,
                         (tot_loss * num_steps_per_update) / num_iter)
                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.
                      format(phase, tot_loc_loss / num_iter,
                             tot_cls_loss / num_iter,
                             (tot_loss * num_steps_per_update) / num_iter))


if __name__ == '__main__':
    # need to add argparse
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='rgb or flow')
    parser.add_argument('-save_model', type=str)

    args = parser.parse_args()
    import experiment.top_30_class as experiment
    train_loader = experiment.dataloader
    val_loader = experiment.val_dataloader
    num_classes = experiment.num_classes
    weight_file = experiment.weight_file
    run(train_loader,
        val_loader,
        num_classes,
        mode="rgb",
        save_model=weight_file,
        log_csv="experiment/top_30_class/train_log.csv")
