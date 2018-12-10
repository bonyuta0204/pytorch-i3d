import os

import torch
from torchvision import transforms

from src.videotransforms import RandomCrop, RandomHorizontalFlip, CenterCrop
from src.mit_data import MITDataset
from src.pytorch_i3d import InceptionI3d

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-2])
print(ROOT_DIR)

INDEX_FILE = os.path.join(ROOT_DIR, "experiment/top_30_class/index.csv")
SPLIT_FILE = os.path.join(ROOT_DIR, "experiment/top_30_class/split.csv")

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

weight_file = "experiment/top_30_class/weight/181128-"

model = InceptionI3d(400, in_channels=3, spatial_squeeze=True)
model.replace_logits(num_classes)
