import os
import pickle
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import videotransforms
from mit_data import MITDataset as Dataset
from mit_data import make_label_binarizer
from pytorch_i3d import InceptionI3d
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms


class Evaluator():
    """
    evaluator = Evaluate(model, dataloader, mlb)
    evaluator.run()
    evaluator.category_accracy(thresh=0.5)
    {"dog": 0.55, "cat": 0.8, ...}
    evaluator.category_auc()
    {"dog": 0.55, "cat": 0.8, ...}
    evaluator[3].true_label()
    ["cat"]
    evaluator[3].sorted_prediction()
    {"dog": 0.99, "cat": 0.33}
    evaluator[4].show_image()
    <Picture of train image>
    """

    def __init__(self, model=None, dataloader=None, mlb=None):
        self.model = model
        self.dataloader = dataloader
        self.result = []
        self.mlb = mlb

    def run(self):
        i = 0
        for data in self.dataloader:
            print(torch.cuda.memory_allocated())
            print("inferencing number {0:4d}".format(i))
            i += 1
            # inputs (N * C * T *... )
            inputs = data["video"]
            # labels (N * num_classes, T)
            labels = data["label"]

            inputs = inputs.cuda()
            labels = labels.cuda()
            # logits (N * num_classes, T)
            squeezed_labels = torch.max(labels, dim=2)[0]
            logits = self.model(inputs)
            squeezed_logits = torch.max(logits, dim=2)[0]
            score = torch.sigmoid(squeezed_logits)

            for n in range(len(inputs)):
                result_dict = {
                    # "video": inputs[n].cpu().numpy(),
                    "label": squeezed_labels[n].cpu().numpy(),
                    "score": score[n].cpu().detach().numpy()
                }
                self.result.append(result_dict)

            # clear memory
            del inputs, labels, logits, squeezed_logits, squeezed_labels, score

    def __getitem__(self, idx):
        return self.result[idx]

    def stats_setup(self):
        """
        make self.label_matrix and self.score_matrix
        each has shape of (N x num_classes)
        """
        label_list = []
        score_list = []
        for sample in self.result:
            label_list.append(sample["label"])
            score_list.append(sample["score"])
        self.label_matrix = np.array(label_list).astype("bool")
        self.score_matrix = np.array(score_list)

    def accuracy(self, thresh=0.5):
        predicted_matrix = self.score_matrix >= thresh
        classify_accuracy_matrix = (predicted_matrix == self.label_matrix)
        return classify_accuracy_matrix.mean(axis=0)

    def auc(self):
        auc_list = []
        for c in range(self.label_matrix.shape[1]):
            category_label = self.label_matrix[:, c]
            category_score = self.score_matrix[:, c]
            auc = roc_auc_score(category_label, category_score)
            auc_list.append(auc)
        return auc_list

    def save_result(self, out_file):
        with open(out_file, "wb") as f:
            pickle.dump(self.result, f)

    def load_result(self, result_file):
        with open(result_file, "rb") as f:
            self.result = pickle.load(f)


if __name__ == "__main__":
    test = Evaluator("", 1, 1)
    test.save_result("test.pkl")
    batch_size = 1
    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(
        mode="train",
        transforms=train_transforms,
        index_file="data/MIT_data/binary_label_man.csv",
        split_file="binary_split.csv")
    print("length of train dataset: {0:4d}".format(len(dataset)))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    val_dataset = Dataset(
        mode="val",
        transforms=test_transforms,
        index_file="data/MIT_data/binary_label_man.csv",
        split_file="binary_split.csv")
    print("length of validation dataset: {0:4d}".format(len(dataset)))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    num_classes = len(dataset.mlb.classes_)

    i3d = InceptionI3d(400, in_channels=3, spatial_squeeze=True)
    i3d.replace_logits(num_classes)
    checkpoint = torch.load("learning_history/binary.pt000400.pt")
    i3d.load_state_dict(checkpoint)
    i3d.cuda()
    # checkpoint = torch.load(
    # "learning_history/binary.pt000400.pt",
    # map_location=torch.device('cpu'))
    # i3d.load_state_dict(checkpoint)
    i3d.eval()
    mlb = dataset.mlb
    evaluator = Evaluator(i3d, val_dataloader, mlb)
    evaluator.run()
    evaluator.save_result("binary_result.pkl")
