import os

import cv2
import numpy as np
from PIL import Image

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-2])


class VideoViewer:
    """
    classes to present videos in some kind of way
    """

    def __init__(self, video_path):
        self.video_path = os.path.join(ROOT_DIR, video_path)
        self.video = None

    def _load_video(self):
        video = []
        cap = cv2.VideoCapture(self.video_path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video.append(frame)
            else:
                break
        cap.release()
        video = np.array(video)
        self.video = video

    def image_as_array(self, step):
        if self.video is None:
            self._load_video()
        return self.video[step]

    def image_as_PIL(self, step):
        if self.video is None:
            self._load_video()
        return Image.fromarray(self.video[step])
