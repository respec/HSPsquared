# content of test_sysexit.py
# this is a very basic pytest file
import pytest


def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()