import os
import pytest

import caffe
import torch
from pytorch_caffe import caffenet

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-2])
MODEL_DEF = os.path.join(ROOT_DIR, "models/bvlc_googlenet/deploy.prototxt")
MODEL_WEIGHT = os.path.join(ROOT_DIR,
                            "models/bvlc_googlenet/bvlc_googlenet.caffemodel")


def test_can_load_googlenet_as_caffe():
    net = caffe.Net(MODEL_DEF, MODEL_WEIGHT, caffe.TEST)
    assert type(net) is caffe._caffe.Net


@pytest.mark.skip(reason="pytorch-caffe is not working")
def test_can_load_googlenet_as_torch():
    net = caffenet.CaffeNet(MODEL_DEF)
    assert type(net) == torch
