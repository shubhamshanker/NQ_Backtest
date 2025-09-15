"""
Parquet Metadata Cache System
============================
Task 9: Build parquet metadata cache for instant access to dataset information
"""

import numpy as np
import pandas as pd
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import pyarrow.parquet as pq


class MetadataCache:
    """Cache parquet metadata for instant access without reading full files."""

    def __init__(self, cache_dir: str = "backtesting/cache"):
        self.cache_dir = Path(cache_dir)
        self.metadata_dir = self.cache_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create_metadata(self, parquet_path: str, force_refresh: bool = False) -> Dict:
        """
        Get cached metadata or create if not exists.

        Args:
            parquet_path: Path to parquet file/directory
            force_refresh: Force refresh of cached metadata

        Returns:
            Dictionary with parquet metadata
        """
        parquet_path = Path(parquet_path)
        cache_key = self._get_cache_key(parquet_path)
        cache_file = self.metadata_dir / f"{cache_key}_meta.json"

        # Check if cache exists and is valid
        if not force_refresh and self._is_cache_valid(cache_file, parquet_path):
            with open(cache_file, 'r') as f:
                cached_meta = json.load(f)
            print(f"✅ Using cached metadata for {parquet_path.name}")
            return cached_meta

        # Create new metadata
        print(f"🔄 Creating metadata cache for {parquet_path.name}...")
        metadata = self._extract_metadata(parquet_path)

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata cached for {parquet_path.name}")
        return metadata

    def _extract_metadata(self, parquet_path: Path) -> Dict:
        """Extract comprehensive metadata from parquet file."""

        try:
            if parquet_path.is_dir():
                # Partitioned dataset
                parquet_file = pq.ParquetDataset(parquet_path)
                schema = parquet_file.schema

                # Get sample for statistics
                sample_table = pq.read_table(parquet_path, columns=['open', 'high', 'low', 'close'])
                sample_df = sample_table.to_pandas()
                total_rows = len(sample_df)

            else:
                # Single file
                parquet_file = pq.ParquetFile(parquet_path)
                schema = parquet_file.schema

                # Read sample
                sample_table = parquet_file.read()
                sample_df = sample_table.to_pandas()
                total_rows = parquet_file.metadata.num_rows

            # Extract time range
            if 'timestamp' in sample_df.columns:
                timestamp_col = 'timestamp'
            elif 'datetime' in sample_df.columns:
                timestamp_col = 'datetime'
            else:
                timestamp_col = None

            if timestamp_col:
                sample_df[timestamp_col] = pd.to_datetime(sample_df[timestamp_col])
                min_time = sample_df[timestamp_col].min()
                max_time = sample_df[timestamp_col].max()
                time_range = [min_time.isoformat(), max_time.isoformat()]
            else:
                time_range = [None, None]

            # Calculate price statistics
            price_stats = {}
            for col in ['open', 'high', 'low', 'close']:
                if col in sample_df.columns:
                    price_stats[col] = {
                        'min': float(sample_df[col].min()),
                        'max': float(sample_df[col].max()),
                        'mean': float(sample_df[col].mean()),
                        'std': float(sample_df[col].std())
                    }

            # Volume statistics
            volume_stats = {}
            if 'volume' in sample_df.columns:
                volume_stats = {
                    'min': float(sample_df['volume'].min()),
                    'max': float(sample_df['volume'].max()),
                    'mean': float(sample_df['volume'].mean())
                }

            # File size
            if parquet_path.is_dir():
                file_size = sum(f.stat().st_size for f in parquet_path.rglob('*.parquet'))
            else:
                file_size = parquet_path.stat().st_size

            metadata = {
                'source_path': str(parquet_path),
                'file_size_mb': file_size / 1024 / 1024,
                'total_rows': total_rows,
                'columns': list(sample_df.columns),
                'time_range': time_range,
                'price_statistics': price_stats,
                'volume_statistics': volume_stats,
                'schema': str(schema),
                'created_at': time.time(),
                'file_hash': self._calculate_file_hash(parquet_path)
            }

            return metadata

        except Exception as e:
            # Fallback minimal metadata
            return {
                'source_path': str(parquet_path),
                'error': str(e),
                'created_at': time.time(),
                'file_hash': self._calculate_file_hash(parquet_path)
            }

    def _calculate_file_hash(self, parquet_path: Path) -> str:
        """Calculate hash of parquet file for change detection."""
        hasher = hashlib.md5()

        if parquet_path.is_dir():
            # Hash directory contents
            for file_path in sorted(parquet_path.rglob('*.parquet')):
                hasher.update(str(file_path.stat().st_mtime).encode())
        else:
            hasher.update(str(parquet_path.stat().st_mtime).encode())

        return hasher.hexdigest()

    def _get_cache_key(self, parquet_path: Path) -> str:
        """Generate unique cache key for parquet path."""
        return str(parquet_path).replace('/', '_').replace('\\', '_').replace('=', '_')

    def _is_cache_valid(self, cache_file: Path, parquet_path: Path) -> bool:
        """Check if cached metadata is still valid."""
        if not cache_file.exists():
            return False

        try:
            with open(cache_file, 'r') as f:
                cached_meta = json.load(f)

            # Check file hash
            current_hash = self._calculate_file_hash(parquet_path)
            return cached_meta.get('file_hash') == current_hash

        except:
            return False

    def list_cached_datasets(self) -> List[Dict]:
        """List all cached datasets."""
        cached_datasets = []

        for cache_file in self.metadata_dir.glob('*_meta.json'):
            try:
                with open(cache_file, 'r') as f:
                    metadata = json.load(f)
                cached_datasets.append(metadata)
            except:
                continue

        return cached_datasets

    def clear_cache(self) -> None:
        """Clear all cached metadata."""
        for cache_file in self.metadata_dir.glob('*_meta.json'):
            cache_file.unlink()
        print("✅ Metadata cache cleared")


if __name__ == "__main__":
    # Test metadata cache
    cache = MetadataCache()

    test_paths = [
        "data_parquet/nq/1min/year=2024",
        "data_parquet/nq/5min/year=2024"
    ]

    for test_path in test_paths:
        if Path(test_path).exists():
            print(f"🧪 Testing metadata cache for {test_path}")

            # First access (should create cache)
            start_time = time.time()
            metadata = cache.get_or_create_metadata(test_path)
            first_time = time.time() - start_time

            # Second access (should use cache)
            start_time = time.time()
            cached_metadata = cache.get_or_create_metadata(test_path)
            second_time = time.time() - start_time

            print(f"   First access: {first_time:.3f}s")
            print(f"   Cached access: {second_time:.3f}s")
            print(f"   Speedup: {first_time/second_time:.1f}x")
            print(f"   Rows: {metadata['total_rows']:,}")
            print(f"   Size: {metadata['file_size_mb']:.1f} MB")