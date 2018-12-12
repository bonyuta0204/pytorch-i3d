import os
import sys
import numpy
import matplotlib.pyplot as plt


def log_init(log_file):
    with open(log_file, "w") as f:
        f.write("step,split,loc_loss,cls_loss,tot_loss\n")


def log_step(log_file, step, split, loc_loss, cls_loss, tot_loss):
    with open(log_file, "a") as f:
        f.write("{},{},{},{},{}\n".format(step, split, loc_loss, cls_loss,
                                          tot_loss))
