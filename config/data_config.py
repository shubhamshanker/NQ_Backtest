"""
Data Configuration Module
=========================
Centralized configuration for data sources and paths.
Supports both CSV and Parquet data sources.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

class DataConfig:
    """Centralized data configuration management."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize data configuration.

        Args:
            config_file: Optional path to JSON config file
        """
        self.base_dir = Path(__file__).parent.parent

        # Default configuration
        self._default_config = {
            "data_sources": {
                "csv": {
                    "enabled": True,
                    "root_path": str(self.base_dir / "data"),
                    "files": {
                        "1min": "NQ_M1_standard.csv",
                        "3min": "NQ_M3.csv",
                        "5min": "NQ_M5.csv",
                        "15min": "NQ_M15.csv"
                    }
                },
                "parquet": {
                    "enabled": True,
                    "root_path": str(self.base_dir / "data_parquet"),
                    "require_migration": True
                }
            },
            "preferences": {
                "default_source": "auto",  # auto, csv, parquet
                "default_symbol": "NQ",
                "default_timeframe": "1min",
                "session_filter": True,
                "timezone": "America/New_York"
            },
            "performance": {
                "chunk_size": 100000,
                "parallel_loading": True,
                "cache_enabled": True
            }
        }

        # Load configuration
        self.config = self._load_config(config_file)

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or environment."""
        config = self._default_config.copy()

        # Load from file if provided
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                config.update(file_config)
            except Exception as e:
                print(f"Warning: Failed to load config file {config_file}: {e}")

        # Override with environment variables
        config = self._apply_env_overrides(config)

        return config

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Data source preferences
        if os.getenv('USE_PARQUET_DATA'):
            parquet_enabled = os.getenv('USE_PARQUET_DATA', '').lower() in ('true', '1', 'yes')
            config['preferences']['default_source'] = 'parquet' if parquet_enabled else 'csv'

        # Paths
        if os.getenv('CSV_DATA_PATH'):
            config['data_sources']['csv']['root_path'] = os.getenv('CSV_DATA_PATH')

        if os.getenv('PARQUET_DATA_PATH'):
            config['data_sources']['parquet']['root_path'] = os.getenv('PARQUET_DATA_PATH')

        # Performance
        if os.getenv('DATA_CHUNK_SIZE'):
            try:
                config['performance']['chunk_size'] = int(os.getenv('DATA_CHUNK_SIZE'))
            except ValueError:
                pass

        return config

    def get_data_path(self, source_type: str, timeframe: Optional[str] = None) -> str:
        """
        Get data path for specified source type and timeframe.

        Args:
            source_type: 'csv' or 'parquet'
            timeframe: Optional timeframe for CSV files

        Returns:
            Path to data
        """
        if source_type == 'csv':
            root_path = self.config['data_sources']['csv']['root_path']
            if timeframe:
                filename = self.config['data_sources']['csv']['files'].get(timeframe)
                if filename:
                    return str(Path(root_path) / filename)
            return root_path
        elif source_type == 'parquet':
            return self.config['data_sources']['parquet']['root_path']
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def get_preferred_source(self) -> str:
        """Get preferred data source based on configuration."""
        preference = self.config['preferences']['default_source']

        if preference == 'auto':
            # Auto-select based on availability
            parquet_enabled = self.config['data_sources']['parquet']['enabled']
            parquet_path = Path(self.config['data_sources']['parquet']['root_path'])
            csv_enabled = self.config['data_sources']['csv']['enabled']

            if parquet_enabled and parquet_path.exists():
                # Check if migration is complete
                nq_dir = parquet_path / 'nq'
                if nq_dir.exists() and any(nq_dir.iterdir()):
                    return 'parquet'

            if csv_enabled:
                return 'csv'

            raise RuntimeError("No data sources available")

        return preference

    def is_source_available(self, source_type: str) -> bool:
        """Check if a data source is available and configured."""
        if source_type not in self.config['data_sources']:
            return False

        source_config = self.config['data_sources'][source_type]
        if not source_config.get('enabled', False):
            return False

        path = Path(source_config['root_path'])
        return path.exists()

    def get_csv_file_path(self, timeframe: str) -> str:
        """Get full path to CSV file for timeframe."""
        root_path = self.config['data_sources']['csv']['root_path']
        filename = self.config['data_sources']['csv']['files'].get(timeframe)

        if not filename:
            raise ValueError(f"No CSV file configured for timeframe: {timeframe}")

        return str(Path(root_path) / filename)

    def save_config(self, output_file: str) -> None:
        """Save current configuration to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save config to {output_file}: {e}")

    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status report."""
        report = {
            'valid': True,
            'issues': [],
            'sources': {}
        }

        # Check CSV source
        csv_config = self.config['data_sources']['csv']
        csv_root = Path(csv_config['root_path'])
        csv_status = {
            'enabled': csv_config['enabled'],
            'root_exists': csv_root.exists(),
            'files_found': {}
        }

        if csv_config['enabled']:
            if not csv_root.exists():
                report['issues'].append(f"CSV root path not found: {csv_root}")
                report['valid'] = False
            else:
                for timeframe, filename in csv_config['files'].items():
                    file_path = csv_root / filename
                    csv_status['files_found'][timeframe] = file_path.exists()
                    if not file_path.exists():
                        report['issues'].append(f"CSV file not found: {file_path}")

        report['sources']['csv'] = csv_status

        # Check Parquet source
        parquet_config = self.config['data_sources']['parquet']
        parquet_root = Path(parquet_config['root_path'])
        parquet_status = {
            'enabled': parquet_config['enabled'],
            'root_exists': parquet_root.exists(),
            'nq_dir_exists': False,
            'timeframes_available': []
        }

        if parquet_config['enabled']:
            if parquet_root.exists():
                nq_dir = parquet_root / 'nq'
                parquet_status['nq_dir_exists'] = nq_dir.exists()
                if nq_dir.exists():
                    parquet_status['timeframes_available'] = [
                        d.name for d in nq_dir.iterdir() if d.is_dir()
                    ]
            else:
                report['issues'].append(f"Parquet root path not found: {parquet_root}")

        report['sources']['parquet'] = parquet_status

        return report

    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config, indent=2)

# Global configuration instance
_config_instance: Optional[DataConfig] = None

def get_data_config() -> DataConfig:
    """Get global data configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = DataConfig()
    return _config_instance

def reload_config(config_file: Optional[str] = None) -> DataConfig:
    """Reload configuration from file."""
    global _config_instance
    _config_instance = DataConfig(config_file)
    return _config_instance

# Convenience functions
def get_data_path(timeframe: str = "1min", source: Optional[str] = None) -> str:
    """
    Get data path for timeframe using current configuration.

    Args:
        timeframe: Data timeframe
        source: Force specific source ('csv' or 'parquet')

    Returns:
        Path to data
    """
    config = get_data_config()
    source = source or config.get_preferred_source()

    if source == 'csv':
        return config.get_csv_file_path(timeframe)
    else:
        return config.get_data_path('parquet')

def is_parquet_available() -> bool:
    """Check if Parquet data is available."""
    config = get_data_config()
    return config.is_source_available('parquet')

def get_preferred_data_source() -> str:
    """Get the preferred data source."""
    config = get_data_config()
    return config.get_preferred_source()

# Example usage
if __name__ == "__main__":
    # Test configuration
    config = DataConfig()

    print("Configuration Status:")
    print("=" * 50)

    # Validate configuration
    report = config.validate_config()
    print(f"Valid: {report['valid']}")

    if report['issues']:
        print("\nIssues:")
        for issue in report['issues']:
            print(f"  - {issue}")

    print(f"\nPreferred source: {config.get_preferred_source()}")

    # Show available sources
    for source_name, source_info in report['sources'].items():
        print(f"\n{source_name.upper()} Source:")
        print(f"  Enabled: {source_info['enabled']}")
        print(f"  Available: {source_info.get('root_exists', False)}")

        if source_name == 'csv' and 'files_found' in source_info:
            print("  Files:")
            for tf, found in source_info['files_found'].items():
                status = "✅" if found else "❌"
                print(f"    {tf}: {status}")

        if source_name == 'parquet' and 'timeframes_available' in source_info:
            print(f"  Timeframes: {source_info['timeframes_available']}")