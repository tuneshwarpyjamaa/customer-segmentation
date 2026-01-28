"""
Customer Segmentation Analysis Package

A lightweight, memory-efficient customer segmentation system using RFM analysis.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data_loader import load_data_optimized
from .data_cleaner import clean_data
from .rfm_analysis import calculate_rfm, segment_customers

__all__ = [
    'load_data_optimized',
    'clean_data',
    'calculate_rfm',
    'segment_customers'
]
