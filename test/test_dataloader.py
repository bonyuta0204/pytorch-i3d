import torch

from src.mit_data import MITDataset


def test_dataset_can_load_data():
    dataset = MITDataset()
    sample = dataset[0]
    video = sample["video"]
    assert video.shape == torch.Size([3, 90, 256,
                                      256]), "data shape is invalid"
    label = sample["label"]
    assert type(label) is torch.Tensor


def test_choose_number_of_frames():
    dataset = MITDataset(frames=32)
    sample = dataset[0]
    video = sample["video"]
    assert video.shape == torch.Size([3, 32, 256,
                                      256]), "data shape is invalid"


def test_dataset_with_normalization():
    dataset = MITDataset(normalize=True)
    sample = dataset[0]
    video = sample["video"]
    assert video.shape == torch.Size([3, 90, 256,
                                      256]), "data shape is invalid"
    label = sample["label"]
    assert type(label) is torch.Tensor
    mean_video = torch.mean(video).item()
    assert abs(
        mean_video) < 5, "mean value is too big. seems not to be normalized"
