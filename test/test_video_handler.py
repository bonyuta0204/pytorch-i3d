from video_handler import VideoViewer
import os
import pytest

@pytest.fixture(scope="module")
def sample_video():
    os.chdir("../")
    return VideoViewer("data/MIT_data/training/aiming/v035_0005.mp4")

def test_video_loaded(sample_video):
    sample_video._load_video()
    assert sample_video.video.shape == (90, 256, 256, 3)
    assert sample_video.image_as_array(5).shape == (256, 256, 3)
