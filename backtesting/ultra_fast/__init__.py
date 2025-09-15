"""
Ultra-Fast Backtesting Engine
============================
Achieving 50x+ speedup through:
- Memory-mapped arrays for zero-copy access
- Numba JIT compilation
- Precomputed indicators
- Parallel execution
- Optimized memory layout
"""

from .data_loader import UltraFastDataLoader
from .numba_engine import NumbaBacktestEngine

__version__ = "1.0.0"
__all__ = ["UltraFastDataLoader", "NumbaBacktestEngine"]