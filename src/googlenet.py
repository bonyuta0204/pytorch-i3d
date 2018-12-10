import torch

import caffe
from pytorch_caffe import caffenet


def forward_pytorch(protofile, weightfile, image):
    """ forward computation over converted pytorch network.

    make pytorch network from caffe protofile and weight file and forward image.

    Args:
        protofile (str): the path of the protofile
        weightfile (str): the path of weightfile
        image (numpy.ndarray): image to pass thorow the network

    Returns:
        Blob: blabla

    """
    net = caffenet.CaffeNet(protofile)
    print(net)
    net.load_weights("models/googlenet/bvlc_googlenet.caffemodel")
    net.eval()
    image = torch.from_numpy(image)
    blobs = net(image)
    return blobs, net.models

if __name__ =="__main__":
    net = caffenet.CaffeNet("models/googlenet/deploy.prototxt")
