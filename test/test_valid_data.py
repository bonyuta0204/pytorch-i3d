import os
import sys

import pandas as pd

from mit_data import MITDataset
from label_handler import LabelHandler

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-1])

print(sys.path)


def test_data_format():
    df = pd.read_csv("data/MIT_data/full-label-index.csv")
    assert "object_label" in df.columns
    assert "hogehoge" not in df.columns


def test_data_accessible_from_classes():
    os.chdir("../")
    dataset = MITDataset(index_file="data/MIT_data/full-label-index.csv")
    assert dataset.index
    label_handler = LabelHandler("data/MIT_data/full-label-index.csv")
    assert label_handler
