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
import torchvision.transforms as transforms
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

from . import videotransforms

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-2])


class MITDataset(Dataset):
    """pytorch Dataset for loading Moments In Time Dataset

    Attributes:
        index (list): lists of dicts. each element in the lists represent a data.
                      each data is python dict which has keys ``index``,
                      ``category``, ``object_label``,  etc.

        mlb: An :py:class:`sklearn.preprocessing.MultiLabelBinarizer` object.
             Used to binarize label.
    """

    def __init__(self,
                 mode="train",
                 split_file="data/MIT_data/split.csv",
                 frames=None,
                 transforms=None,
                 filter_func=None,
                 normalize=False,
                 expand_label=False,
                 index_file="data/MIT_data/full-label-index.csv"):
        """ Initialize MITDataset

        Args:
            mode (str): `train`, `val`, or `test`.
            split_file (str): path to split_file.
            frames (int): number of frames to use from original video
            transformes: torchvision transform object
            filter_func (function): function to filter each data.
            index_file (str): index file to use.
        """
        self.mode = mode
        self.transforms = transforms
        self.split_file = os.path.join(ROOT_DIR, split_file)
        self.frames = frames
        self.expand_label = expand_label
        self.normalize = normalize

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
        returns the length of the dataset

        *Example*

        >>> dataset = MITDataset()
        >>> len(dataset)
        840
        """
        return len(self.index)

    def __getitem__(self, idx):
        """
        returns (video_tensor, label_vector)
        video_tensor is a torch.FloatTensor of shape (C x T x H x W)

        *Example*

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
        video_array = self.load_video(os.path.join(ROOT_DIR, video_path))

        if self.transforms:
            video_array = self.transforms(video_array)
        if self.normalize:
            array_to_normalized_tensor = transforms.Compose([
                videotransforms.ToTensor(),
                videotransforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            video_tensor = array_to_normalized_tensor(video_array)
        else:
            video_tensor = torch.from_numpy(
                video_array.transpose([3, 0, 1, 2]))
            video_tensor = torch.as_tensor(video_tensor, dtype=torch.float)
        label = self.binary_label[idx]

        if self.expand_label:
            label = [label for _ in range(len(video_array))]
            label = np.stack(label, axis=1)

        label_tensor = torch.from_numpy(label)
        label_tensor = torch.as_tensor(label_tensor, dtype=torch.float)

        item["video"] = video_tensor
        item["label"] = label_tensor
        item["video_path"] = video_path
        return item

    def load_video(self, video_path):
        """ load video in ``video_path`` and returns numpy.ndarray

        Args:
            video_path (srt): path to video

        Return:
            numpy.ndarray: numpy array of size (T x W x H x C)

        """
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
        video_length = len(video)
        if self.frames is not None:
            frame_sample_rate = video_length // self.frames
            sample_frame = np.arange(self.frames) * frame_sample_rate
            video = video[sample_frame]
        return video


class MITImageDataset(Dataset):
    """pytorch Dataset for loading Moments In Time Dataset as 2DImage

    Attributes:
        index (list): lists of dicts. each element in the lists represent a data.
                      each data is python dict which has keys ``index``,
                      ``category``, ``object_label``,  etc.

        mlb: An :py:class:`sklearn.preprocessing.MultiLabelBinarizer` object.
             Used to binarize label.
    """

    def __init__(self,
                 mode="train",
                 split_file="data/MIT_data/split.csv",
                 frames=None,
                 transforms=None,
                 normalize=False,
                 filter_func=None,
                 expand_label=False,
                 index_file="data/MIT_data/full-label-index.csv"):
        """ Initialize MITDataset

        Args:
            mode (str): `train`, `val`, or `test`.
            split_file (str): path to split_file.
            frames (int): number of frames to use from original video
            transformes: torchvision transform object. transform should return
                         transformed image either in numpy.ndarray of PIL Image
                         instance
            filter_func (function): function to filter each data.
            index_file (str): index file to use.
        """
        self.mode = mode
        self.transforms = transforms
        self.split_file = os.path.join(ROOT_DIR, split_file)
        self.frames = frames
        self.expand_label = expand_label
        self.normalize = normalize

        if self.frames is None:
            self.frames = 90

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
        returns the length of the dataset

        *Example*

        >>> dataset = MITImageDataset()
        >>> len(dataset)
        75600
        """
        return len(self.index) * self.frames

    def __getitem__(self, idx):
        """
        returns (image_tensor, label_vector)
        image_tensor is a torch.FloatTensor of shape (C x H x W)

        *Example*

        >>> dataset = MITImageDataset()
        >>> sample = dataset[0]
        >>> image = sample["video"]
        >>> image.shape
        torch.Size([3, 256, 256])
        >>> label = sample["label"]
        >>> type(label)
        <class 'torch.Tensor'>
        """
        # select frame F from Video V
        video_idx = int(idx) // self.frames
        frame_idx = int(idx) % self.frames
        item = {}
        row = self.index[video_idx]
        video_path = os.path.join(row["directory"][3:], row["filename"])
        video_array = self.load_video(os.path.join(ROOT_DIR, video_path))
        image_array = video_array[frame_idx]

        if self.transforms:
            # image_array can be numpy.ndarray or PIL Image
            image_array = self.transforms(image_array)
        if self.normalize:
            array_to_normalized_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = array_to_normalized_tensor(image_array)
        else:
            image_tensor = torch.from_numpy(image_array.transpose([2, 0, 1]))
            image_tensor = torch.as_tensor(image_tensor, dtype=torch.float)
        label = self.binary_label[video_idx]
        label_tensor = torch.from_numpy(label)
        label_tensor = torch.as_tensor(label_tensor, dtype=torch.float)

        # the label is kept to "video" for consistency with video dataset
        item["video"] = image_tensor
        item["label"] = label_tensor
        item["video_path"] = video_path
        return item

    def load_video(self, video_path):
        """ load video in ``video_path`` and returns numpy.ndarray

        Args:
            video_path (srt): path to video

        Return:
            numpy.ndarray: numpy array of size (T x W x H x C)

        """
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
        video_length = len(video)
        if self.frames is not None:
            frame_sample_rate = video_length // self.frames
            sample_frame = np.arange(self.frames) * frame_sample_rate
            video = video[sample_frame]
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
