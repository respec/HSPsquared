
import pytest
import os

def test_h5_file_exists():
    assert os.path.exists('test10.h5')


def test_that_should_fail():
    assert os.path.exists('test100000.h5')
