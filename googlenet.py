import torch

import caffe
from pytorch_caffe import caffenet


def forward_pytorch(protofile, weightfile, image):
    net = caffenet.CaffeNet(protofile)
    print(net)
    net.load_weights("models/googlenet/bvlc_googlenet.caffemodel")
    net.eval()
    image = torch.from_numpy(image)
    blobs = net(image)
    return blobs, net.models

if __name__ =="__main__":
    net = caffenet.CaffeNet("models/googlenet/deploy.prototxt")
