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
