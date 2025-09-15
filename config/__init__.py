"""
Configuration Module
===================
Centralized configuration for the trading system.
"""

from .data_config import (
    get_data_config,
    get_data_path,
    is_parquet_available,
    get_preferred_data_source
)

__all__ = [
    'get_data_config',
    'get_data_path',
    'is_parquet_available',
    'get_preferred_data_source'
]
