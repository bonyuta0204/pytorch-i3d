import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, transforms


def log_init(log_file):
    if os.path.exists(log_file):
        raise RuntimeError(
            "{} already exist. Overwriting log file is prohibited".format(
                log_file))
    with open(log_file, "w") as f:
        f.write("step,split,loss\n")


def log_step(log_file, step, split, loss):
    with open(log_file, "a") as f:
        f.write("{},{},{}\n".format(step, split, loss))


def train(train_loader,
          val_dataloader,
          model,
          init_lr=0.1,
          max_steps=64e3,
          num_steps=4,
          save_model_dir="",
          device=torch.device("cuda:0"),
          loss_func=nn.BCEWithLogitsLoss(),
          log_file="log.csv",
          save_steps=100,
          val_steps=100):

    lr = init_lr
    model.train(True)
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    steps = 0
    log_init(log_file)

    # train it
    while steps < max_steps:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)
        # Each epoch has a training and validation phase
        num_iter = 0
        optimizer.zero_grad()
        cum_loss = 0
        step_loss = 0

        # Iterate over data.
        for data in train_loader:
            num_iter += 1

            # get the inputs
            inputs = data["video"]
            labels = data["label"]

            # compute the logits
            inputs = Variable(inputs.to(device=device))
            labels = Variable(labels.to(device=device))
            logits = model(inputs)

            loss = loss_func(logits, labels)
            cum_loss += loss.data.item()
            step_loss += loss.data.item()
            loss.backward()

            if num_iter % num_steps == 0:
                steps += 1
                # validation
                if steps % val_steps == 1:
                    print("computing validation loss...")
                    val_loss = validation_loss(
                        val_dataloader, model, loss_func, device=device)
                    log_step(log_file, steps, "val", val_loss)
                    print('validation error step: {0:4d} loss: {1:.4f}'.format(
                        steps, val_loss))
                optimizer.step()
                optimizer.zero_grad()
                lr_sched.step()
                log_step(log_file, steps, "train", step_loss / num_steps)
                step_loss = 0
                if steps % 10 == 0:
                    print('step: {0:4d} loss: {1:.4f}'.format(
                        steps, cum_loss / (10 * num_steps)))
                    # save model
                    cum_loss = 0
            if num_steps % save_steps == 0:
                save_file = os.path.join(save_model_dir,
                                         "{0:06d}.pt".format(steps))
                torch.save(model.state_dict(), save_file)


def validation_loss(val_dataloader,
                    model,
                    loss_func,
                    device=torch.device("cuda:0")):
    cum_loss = 0
    model.train(False)
    for data in val_dataloader:
        video = data["video"]
        label = data["label"]
        video = video.to(device=device)
        label = label.to(device=device)
        logit = model(video)
        loss_func = nn.BCEWithLogitsLoss()
        loss = loss_func(logit, label)
        cum_loss += loss.item()
    model.train(True)
    return cum_loss / len(val_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, help='rgb or flow')
    parser.add_argument('-save_model', type=str)
