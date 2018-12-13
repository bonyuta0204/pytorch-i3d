import os

import pytest

from src.train import log_init


def test_lof_file_cannot_be_overwritten():
    log_file_name = "temp.log"
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    log_init(log_file_name)
    with pytest.raises(RuntimeError):
        log_init(log_file_name)
