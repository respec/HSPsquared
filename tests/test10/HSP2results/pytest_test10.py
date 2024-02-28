
import pytest
import os

def test_h5_file_exists():
    assert os.path.exists('test10.h5')