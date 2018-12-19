import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

from .label_handler import LabelHandler


class ResultAnalyzer():
    """ Class to analyse evaluated result

        Args
        make self.label_matrix and self.score_matrix
        each has shape of (N x num_classes)
    """

    def __init__(self, result_file, index_file):
        self.index_file = index_file
        with open(result_file, "rb") as f:
            self.result = pickle.load(f)
        label_list = []
        score_list = []
        for sample in self.result:
            label_list.append(sample["label"])
            score_list.append(sample["score"])
        self.label_matrix = np.array(label_list).astype("bool")
        self.score_matrix = np.array(score_list)

    def __getitem__(self, idx):
        return self.result[idx]

    def accuracy(self, thresh=0.5):
        predicted_matrix = self.score_matrix >= thresh
        classify_accuracy_matrix = (predicted_matrix == self.label_matrix)
        return classify_accuracy_matrix.mean(axis=0)

    def auc(self):
        auc_list = []
        for c in range(self.label_matrix.shape[1]):
            category_label = self.label_matrix[:, c]
            category_score = self.score_matrix[:, c]
            try:
                auc = roc_auc_score(category_label, category_score)
            except:
                auc = None
            auc_list.append(auc)
        return auc_list

    def plot_auc(self, save_file, title="AUC", figsize=(12, 8)):
        auc = self.auc()
        lh = LabelHandler(self.index_file)
        lh.expand_index()
        label_count = lh.label_count()
        classes = make_label_binarizer(self.index_file).classes_

        barplot_before_sort = zip(classes, auc)
        barplot_sorted = sorted(
            barplot_before_sort,
            key=lambda data: label_count[data[0]],
            reverse=True)

        x = np.arange(len(barplot_sorted)) * 1.2
        labels = list(map(lambda d: d[0], barplot_sorted))
        auc = np.array(list(map(lambda d: d[1],
                                barplot_sorted))).astype(np.float)

        plt.figure(figsize=figsize)
        plt.bar(x, auc, label="AUC")
        plt.xticks(x, labels)
        plt.legend()
        plt.title(title)
        plt.savefig(save_file)


def plot_learning_curve(log_file,
                        save_file,
                        figsize=(12, 8),
                        title="learning curve"):
    df_loss = pd.read_csv(log_file)
    train_loss = df_loss[df_loss["split"] == "train"]
    val_loss = df_loss[df_loss["split"] == "val"]
    plt.figure(figsize=figsize)
    plt.plot(
        train_loss["step"].values,
        train_loss["loss"].values,
        label="train loss")
    plt.plot(
        val_loss["step"].values,
        val_loss["loss"].values,
        label="validation loss")
    plt.legend()
    plt.title(title)


def make_label_binarizer(index_file):
    train_index = pd.read_csv(index_file)
    labels = train_index["object_label"]
    labels = list(map(lambda data: str(data).split(" "), labels))
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    return mlb
