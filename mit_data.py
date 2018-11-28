import json
import os
import random
# Ignore warnings
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-1])


class MITDataset(Dataset):
    """MIT Data Set"""

    def __init__(self,
                 mode="train",
                 split_file="train_index.json",
                 transforms=None,
                 filter_func=None,
                 index_file="data/MIT_data/full-label-index.csv"):
        self.mode = mode
        self.transforms = transforms
        self.split_file = os.path.join(ROOT_DIR, split_file)

        with open(self.split_file, "r") as f:
            split = json.load(f)

        df = pd.read_csv(os.path.join(ROOT_DIR, index_file), index_col="index")

        if (mode == "train" or mode == "val"):
            df = df[df["split"] == "train"]
            df = df.iloc[split[mode]]
            self.index = df.to_dict("records")

        elif mode == "test":
            df = df[df["split"] == "test"]
            self.index = df.to_dict("records")

        for data in self.index:
            data["object_label_list"] = str(data["object_label"]).split(" ")

        if filter_func is not None:
            self.index = list(
                filter(lambda data: filter_func(data), self.index))

        self._i = 0
        self.mlb = make_label_binarizer(index_file)
        labels = list(
            map(lambda data: str(data["object_label"]).split(" "), self.index))
        self.binary_label = self.mlb.transform(labels)

    def __len__(self):
        """
        Exaple
        ---------------
        >>> dataset = MITDataset()
        >>> len(dataset)
        840
        """
        return len(self.index)

    def __getitem__(self, idx):
        """
        returns (video_tensor, label_vector)
        video_tensor is a torch.FloatTensor of shape (C x T x H x W)

        Example
        --------------------------------
        >>> mlb = make_label_binarizer("data/MIT_data/train_index.csv")
        >>> dataset = MITDataset()
        >>> sample = dataset[0]
        >>> video = sample["video"]
        >>> video.shape
        torch.Size([3, 90, 256, 256])
        >>> label = sample["label"]
        >>> type(label)
        <class 'torch.Tensor'>
        """
        item = {}
        row = self.index[idx]
        video_path = os.path.join(row["directory"][3:], row["filename"])
        video_array = self.load_video(video_path)
        if self.transforms:
            video_array = self.transforms(video_array)
        video_tensor = torch.from_numpy(video_array.transpose([3, 0, 1, 2]))
        video_tensor = torch.as_tensor(video_tensor, dtype=torch.float)
        label = self.binary_label[idx]
        label = [label for _ in range(len(video_array))]
        label = np.stack(label, axis=1)
        label_tensor = torch.from_numpy(label)
        label_tensor = torch.as_tensor(label_tensor, dtype=torch.float)

        item["video"] = video_tensor
        item["label"] = label_tensor
        return item

    def load_video(self, video_path):
        video = []
        cap = cv2.VideoCapture(video_path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video.append(frame)
            else:
                break
        cap.release()
        video = np.array(video)

        return video


def train_val_split(length, val_size=0.3, save_file="train_index.json"):
    """
    return index of data for using split
    return index of data in train_set
    """
    split_dict = {}
    index_set = set(range(length))
    val_set = set(random.sample(range(length), round(length * val_size)))
    train_set = index_set - val_set
    split_dict["train"] = list(train_set)
    split_dict["val"] = list(val_set)
    with open(save_file, "w") as f:
        json.dump(split_dict, f)


def make_label_binarizer(index_file):
    index_file = os.path.join(ROOT_DIR, index_file)
    train_index = pd.read_csv(index_file)
    labels = train_index["object_label"]
    labels = list(map(lambda data: str(data).split(" "), labels))
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    return mlb


if __name__ == "__main__":
    mlb = make_label_binarizer("data/MIT_data/train_index.csv")
    dataset = MITDataset(mlb)
    sample = dataset[0]
    video = sample["video"]
    train_val_split(1200)
