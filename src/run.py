import os


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
