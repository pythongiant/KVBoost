"""
pytest configuration for the kvboost test suite.

Markers:
  slow   — tests that require a model download or GPU. Run with: pytest -m slow
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as requiring model download or GPU")
