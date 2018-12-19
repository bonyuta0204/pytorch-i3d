import os
import pickle

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class Evaluator():
    # TODO: separate resulet analyser from this module
    """ Class to evaluate CNN model

    This class has two function

    * evaluate validaion or tested data over trained model and store the result
    * compute some basic statics value

    Attributes:
        mlb: : ``MultiLabelBinarizer`` used to convert binary_label

    *Example*

    >>> evaluator = Evaluate(model, dataloader, mlb)
    >>> evaluator.run()
    """

    def __init__(self, mlb=None):
        self.result = []
        self.mlb = mlb

    def run(self,
            model,
            dataloader,
            result_file,
            device=torch.device("cuda:0")):
        """ evaluate validation or test over trained model

        Args:
            model: torch model to evaluate. It should output logits as final
                   output.
            dataloader: pytorch dataloader. MITDataset or MITImageDataset class
                        object is supported.
            result_file (str): file to write result. Cannot be overwritten.
        Returns:
            list: each dict represents output for a single input.
                  each dict has three keys ``video``, ``label``,
                  ``score`` respectedly has path of video file,
                  binary label, and row logits value.
        """
        if os.path.exists(result_file):
            raise RuntimeError(
                "{} already exist. Overwriting evaluation file is prohibited".
                format(result_file))

        i = 0
        for data in dataloader:
            print(torch.cuda.memory_allocated())
            print("inferencing number {0:4d}".format(i))
            i += 1
            # inputs (N * C * T *... )
            inputs = data["video"]
            # labels (N * num_classes, T)
            labels = data["label"]

            # inputs, labels = data

            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            # logits (N * num_classes, T)
            logits = model(inputs)
            score = torch.sigmoid(logits)

            for n in range(len(inputs)):
                result_dict = {
                    "video": data["video_path"][n],
                    "label": labels[n].cpu().numpy(),
                    "score": score[n].cpu().detach().numpy()
                }
                self.result.append(result_dict)

            # clear memory
            del inputs, labels, logits, score
            print("saving result to {}".format(result_file))
            self._save_result(result_file)
        return self.result

    def __getitem__(self, idx):
        return self.result[idx]

    def _save_result(self, out_file):
        with open(out_file, "wb") as f:
            pickle.dump(self.result, f)
