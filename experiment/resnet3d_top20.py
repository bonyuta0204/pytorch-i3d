import os

import torch
from torchvision import transforms

from src.mit_data import MITDataset
from src.pytorch_i3d import InceptionI3d
from src.videotransforms import CenterCrop, RandomCrop, RandomHorizontalFlip

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-2])
