import copy
import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from src.i3res import I3ResNet
from src.mit_data import MITDataset
from src.videotransforms import CenterCrop, RandomCrop, RandomHorizontalFlip

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-2])

INDEX_FILE = os.path.join(ROOT_DIR, "experiment/binary_class/binary_class.csv")
SPLIT_FILE = os.path.join(ROOT_DIR, "experiment/binary_class/split.csv")
NUM_FRAMES = 32

batch_size = 1
train_transforms = transforms.Compose([
    RandomCrop(224),
    RandomHorizontalFlip(),
])
test_transforms = transforms.Compose([CenterCrop(224)])

dataset = MITDataset(
    mode="train",
    transforms=train_transforms,
    index_file=INDEX_FILE,
    normalize=True,
    frames=NUM_FRAMES,
    split_file=SPLIT_FILE)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10,
    pin_memory=True)

val_dataset = MITDataset(
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
