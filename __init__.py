"""
py-occam: Python implementation of OCCAM variable-based reconstructability analysis
"""

__version__ = "0.1.0"

from .manager import Occam
from .utils import load_data

__all__ = ["Occam", "load_data"]