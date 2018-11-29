import os
import matplotlib.pyplot as plt

import cv2
import numpy as np
from PIL import Image

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-1])


class VideoViewer:
    """
    classes to present videos in some kind of way
    """


def crop_image(self, frames=[22, 45, 67]):
    if not os.path.isdir(self.image_dir):
        os.mkdir(self.image_dir)

    infile = os.path.join(self.dir_path, self.filename)

    video = []
    cap = cv2.VideoCapture(infile)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    cap.release()

    for frame in frames:
        img_path = os.path.join(self.image_dir, "{0:04d}.png".format(frame))
        if os.path.isfile(img_path):
            continue
        img = video[frame]
        img = Image.fromarray(np.uint8(img))
        img.save(img_path)


def load_images(self, frames=[22, 45, 67], type="imagenet"):
    paths = []
    for frame in frames:
        img_path = os.path.join(self.image_dir, "{0:04d}.png".format(frame))
        paths.append(img_path)

    return paths


def show_images(self):
    images = self.load_images()
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images)
    for i in range(n_images):
        img = np.asarray(Image.open(images[i]))
        axes[i].imshow(img)
        axes[i].axis('off')
    return fig, axes
