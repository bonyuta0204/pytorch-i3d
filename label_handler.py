import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-1])


class LabelHandler():
    """
    class for handling label data
    function include
    * show label stats
    * convert binary label to named label
    etc.
    >>> lh = LabelHandler("data/MIT_data/full-label-index.csv")
    """

    def __init__(self, index_file):
        self.index = pd.read_csv(os.path.join(ROOT_DIR, index_file))

    def expand_index(self):
        """
        expand index table and make new expanded table in which video
        has only one label.

        expaned label can be acquired as self.expanded_df
        >>> lh = LabelHandler("data/MIT_data/full-label-index.csv")
        >>> lh.expand_index()
        """
        labels = self.index["object_label"].map(lambda x: str(x).split(" "))
        expanded = []
        for n in range(self.index.shape[0]):
            for i in range(len(labels[n])):
                dict_element = self.index.iloc[n].to_dict()
                dict_element["object_label"] = labels[n][i]
                dict_element["index"] = n
                expanded.append(dict_element)
        expanded_df = pd.DataFrame(expanded)
        expanded_df["object_label"] = expanded_df["object_label"].map(
            lambda x: np.nan if x == 'nan' else x)
        self.expanded_df = expanded_df

    def sorted_label(self, ascending=False):
        """
        >>> lh = LabelHandler("data/MIT_data/full-label-index.csv")
        >>> lh.expand_index()
        >>> lh.sorted_label()
        """
        count = self.expanded_df[["object_label",
                                  "index"]].groupby(["object_label"],
                                                    as_index=True).count()
        count = count.to_dict()["index"]
        ordered_count = OrderedDict(
            sorted(count.items(), key=lambda t: t[1], reverse=(not ascending)))
        return ordered_count

    def label_count(self):
        count = self.expanded_df[["object_label",
                                  "index"]].groupby(["object_label"],
                                                    as_index=True).count()
        count = count.to_dict()["index"]
        return count
