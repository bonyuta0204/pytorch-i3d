import copy
import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from src.i3res import I3ResNet
from src.mit_data import MITImageDataset

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-2])

INDEX_FILE = os.path.join(ROOT_DIR, "experiment/top_30_class/index.csv")
SPLIT_FILE = os.path.join(ROOT_DIR, "experiment/top_30_class/split.csv")
NUM_FRAMES = 32

batch_size = 1
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(224),
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(224),
])

dataset = MITImageDataset(
    mode="train",
    transforms=train_transforms,
    normalize=True,
    index_file=INDEX_FILE,
    frames=NUM_FRAMES,
    split_file=SPLIT_FILE)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10,
    pin_memory=True)

val_dataset = MITImageDataset(
    mode="val",
    transforms=test_transforms,
    frames=NUM_FRAMES,
    normalize=True,
    index_file=INDEX_FILE,
    split_file=SPLIT_FILE)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=10,
    pin_memory=True)

mlb = dataset.mlb
num_classes = len(dataset.mlb.classes_)

resnet = torchvision.models.resnet50(pretrained=True)
resnet.fc = nn.Linear(2048, num_classes)
model = I3ResNet(copy.deepcopy(resnet), NUM_FRAMES)
