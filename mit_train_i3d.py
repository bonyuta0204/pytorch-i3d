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
from torchvision import transforms

import videotransforms
from mit_data import MITDataset as Dataset
from mit_data import make_label_binarizer
from pytorch_i3d import InceptionI3d

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument(
    '-save_model', type=str, default="learning_history/train.pt")
args = parser.parse_args()

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-1])


def filter_label(data, label):
    return len(
        data["object_label_list"]) == 1 and label in data["object_label_list"]

def filter_man(data):
    return filter_label(data, "man")


def run(init_lr=0.1, max_steps=64e3, mode='rgb', batch_size=2, save_model=''):
    # setup dataset
    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(
        mode="train",
        transforms=train_transforms,
        index_file="data/MIT_data/binary_label_man.csv",
        split_file="binary_split.csv")
    print("length of train dataset: {0:4d}".format(len(dataset)))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=36,
        pin_memory=True)

    val_dataset = Dataset(
        mode="val",
        transforms=test_transforms,
        index_file="data/MIT_data/binary_label_man.csv",
        split_file="binary_split.csv")
    print("length of validation dataset: {0:4d}".format(len(dataset)))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=36,
        pin_memory=True)

    num_classes = len(dataset.mlb.classes_)
    print(num_classes)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

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
                # inputs, labels = data
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

                # compute localization loss
                # TODO size is wrong here
                # loc_loss = F.binary_cross_entropy_with_logits(
                #    per_frame_logits, labels)
                #tot_loc_loss += loc_loss.data[0]

                # compute classification loss
                # (with max-pooling along time B x C x T)
                # cls_loss = F.binary_cross_entropy_with_logits(
                #    torch.max(per_frame_logits, dim=2)[0],
                #    torch.max(labels, dim=2)[0])
                #tot_cls_loss += cls_loss.data[0]

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
                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.
                      format(phase, tot_loc_loss / num_iter,
                             tot_cls_loss / num_iter,
                             (tot_loss * num_steps_per_update) / num_iter))


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, save_model=args.save_model)
