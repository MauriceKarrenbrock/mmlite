# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
"""Tests."""
from pathlib import Path


def base_dir():
    """Path to the test dir"""
    return Path(__file__).parent.absolute()


def test_0():
    """Test template."""
    assert bool(1) is True
