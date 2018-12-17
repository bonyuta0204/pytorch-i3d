import os
import re


class Run(object):
    """ Class to manage each training and evaluation run.

    This class manage each run. the main function is to handle the directory
    for logging, saving weights, and so on.
    example layout would become something as::

        root_dir/ -- train_log.csv
                 L- weights/ -- 000100.pt
                             L- 000200.pt

    Attributes:
        root_dir (str): root directory for run. if not existed, automatically
                        created in recursive way (like ``mkdir -p``))
    Args:
        root_dir (str): root directory for run. if not existed, automatically
                        created in recursive way (like ``mkdir -p``))
        log_file (str): path for log file.
        weights_dir (str): path for saved weights directory

    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        if not os.path.isdir(self.root_dir):
            os.makedirs(self.root_dir)
            print("{} is created".format(self.root_dir))
        self.log_file = os.path.join(self.root_dir, "train_log.csv")
        self.weights_dir = os.path.join(self.root_dir, "weights")
        if not os.path.isdir(self.weights_dir):
            os.makedirs(self.weights_dir)
            print("{} is created".format(self.weights_dir))
        self.train_eval = os.path.join(self.root_dir, "train_eval.csv")
        self.val_eval = os.path.join(self.root_dir, "val_eval.csv")

    def figure_file(self, figname):
        """ return figure filename

        """
        figdir = os.path.join(self.root_dir, "figure")
        if not os.path.isdir(figdir):
            os.makedirs(figdir)
        return os.path.join(figdir, figname)

    def weight_step(self, steps):
        """ file name for weight for given step

        """
        return os.path.join(self.weights_dir, "{0:06d}.pt".format(steps))

    def train_eval_step(self, steps):
        return os.path.join(self.root_dir,
                            "train_eval_{0:06d}.csv".format(steps))

    def val_eval_step(self, steps):
        return os.path.join(self.root_dir,
                            "val_eval_{0:06d}.csv".format(steps))

    def weights_available_step(self):
        weights_files = os.listdir(self.weights_dir)
        available_steps = []
        for weights_file in weights_files:
            numeric = re.match(r"[0-9]+", weights_file)
            available_steps.append(int(numeric.group()))
        return available_steps

