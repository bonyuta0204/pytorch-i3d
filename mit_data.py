

import os
# Ignore warnings
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-1])


class MITDataset(Dataset):
    """MIT Data Set"""

    def __init__(self,
                 root_dir=os.path.join(ROOT_DIR, "data/MIT_data"),
                 train=True):
        self.train = train
        self.root_dir = root_dir

        if self.train:
            self.data_dir = os.path.join(root_dir, "training")
        else:
            self.data_dir = os.path.join(root_dir, "test")
        try:
            if self.train:
                self.index = pd.read_csv(
                    os.path.join(self.root_dir, "train_index.csv"),
                    index_col="index")
            else:
                self.index = pd.read_csv(
                    os.path.join(self.root_dir, "test_index.csv"),
                    index_col="index")
        except FileNotFoundError:
            print("parseing directories")
            self.parse_directory()
            self.index = pd.read_csv(
                os.path.join(self.root_dir, "test_index.csv"),
                index_col="index")

        self._i = 0

    def __len__(self):
        """
        Exaple
        ---------------
        >>> dataset = MITDataset()
        >>> len(dataset)
        1200
        """
        return len(self.index)

    def __getitem__(self, idx):
        """
        returns torch.FloatTensor of shape (C x T x H x W)

        Example
        --------------------------------
        >>> dataset = MITDataset()
        >>> sample = dataset[0]
        >>> video = sample["video"]
        >>> video.shape
        torch.Size([3, 90, 256, 256])
        """

        item = {}
        row = self.index.iloc[idx]
        video_path = os.path.join(row["directory"][3:], row["filename"])
        video_array = self.load_video(video_path)
        video_tensor = torch.from_numpy(video_array.transpose([3, 0, 1, 2]))
        item["video"] = video_tensor
        return item

    def load_video(self, video_path):
        video = []
        cap = cv2.VideoCapture(video_path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                video.append(frame)
            else:
                break
        cap.release()
        video = np.array(video)

        return video


if __name__ == "__main__":
    dataset = MITDataset()
    sample = dataset[0]
    video = sample["video"]
    print(video.shape)
