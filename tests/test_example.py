"""Example test file"""
import pytest

def test_basic():
    """Basic test to ensure pytest works"""
    assert 1 + 1 == 2

def test_import():
    """Test that we can import main modules"""
    try:
        import numpy as np
        import pandas as pd
        import torch
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
