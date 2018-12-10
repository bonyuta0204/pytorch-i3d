import os

import pytest

from src.video_handler import VideoViewer


@pytest.fixture(scope="module")
def sample_video():
    os.chdir("../")
    return VideoViewer("data/MIT_data/training/aiming/v035_0005.mp4")


def test_video_loaded(sample_video):
    assert os.path.isfile(sample_video.video_path), "{} does not exist".format(
        sample_video.video_path)
    sample_video._load_video()
    assert sample_video.video.shape == (90, 256, 256, 3)
    assert sample_video.image_as_array(5).shape == (256, 256, 3)
