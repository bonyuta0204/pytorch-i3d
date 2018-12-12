import pickle

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class Evaluator():
    """
    *Example*

    >>> evaluator = Evaluate(model, dataloader, mlb)
    >>> evaluator.run()
    >>> evaluator.category_accracy(thresh=0.5)
    {"dog": 0.55, "cat": 0.8, ...}
    >>> evaluator.category_auc()
    {"dog": 0.55, "cat": 0.8, ...}
    >>> evaluator[3].true_label()
    ["cat"]
    >>> evaluator[3].sorted_prediction()
    {"dog": 0.99, "cat": 0.33}
    >>> evaluator[4].show_image()
    <Picture of train image>
    """

    def __init__(self, mlb=None):
        self.result = []
        self.mlb = mlb

    def run(self, model, dataloader):
        i = 0
        for data in dataloader:
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
            logits = model(inputs)
            squeezed_logits = torch.max(logits, dim=2)[0]
            score = torch.sigmoid(squeezed_logits)

            for n in range(len(inputs)):
                result_dict = {
                    "video": data["video_path"][n],
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
            try:
                auc = roc_auc_score(category_label, category_score)
            except:
                auc = None
            auc_list.append(auc)
        return auc_list

    def save_result(self, out_file):
        with open(out_file, "wb") as f:
            pickle.dump(self.result, f)

    def load_result(self, result_file):
        with open(result_file, "rb") as f:
            self.result = pickle.load(f)


if __name__ == "__main__":
    import experiment.top_30_class as experiment

    dataset = experiment.dataset
    val_dataset = experiment.val_dataset
    dataloader = experiment.dataloader
    val_dataloader = experiment.val_dataloader
    i3d = experiment.model

    checkpoint = torch.load("experiment/top_30_class/weight/181128-004750.pt")
    i3d.load_state_dict(checkpoint)
    i3d.cuda()
    i3d.eval()

    mlb = experiment.mlb
    evaluator = Evaluator(mlb)
    evaluator.run(i3d, val_dataloader)
    evaluator.save_result("experiment/top_30_class/result_with_video.pkl")