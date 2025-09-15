#!/usr/bin/env python3
"""
Data Path Refactoring Script
============================
Automatically refactors hardcoded CSV paths across the codebase
to use the new centralized data configuration system.
"""

import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPathRefactor:
    """Refactors hardcoded data paths to use centralized configuration."""

    def __init__(self, project_root: str = "/Users/shubhamshanker/bt_"):
        self.project_root = Path(project_root)
        self.changes_made = []

        # Patterns to find hardcoded paths
        self.path_patterns = [
            # Direct CSV file paths
            r'["\']([^"\']*\.csv)["\']',
            # Path with data directory
            r'["\']([^"\']*data[^"\']*\.csv)["\']',
            # NQ data files specifically
            r'["\']([^"\']*NQ_M\d+[^"\']*\.csv)["\']',
            # Trading dashboard paths
            r'["\']([^"\']*trading_dashboard[^"\']*\.csv)["\']'
        ]

        # Known CSV files to replace
        self.csv_mappings = {
            'NQ_M1_standard.csv': '1min',
            'NQ_M3.csv': '3min',
            'NQ_M5.csv': '5min',
            'NQ_M15.csv': '15min',
            'NQ_M1.csv': '1min'
        }

        # Files to process
        self.target_files = []
        self._find_target_files()

    def _find_target_files(self) -> None:
        """Find all Python files that might contain hardcoded paths."""
        python_files = list(self.project_root.rglob("*.py"))

        # Filter out files we don't want to modify
        exclude_patterns = [
            '__pycache__',
            '.git',
            'venv',
            'env',
            '.pytest_cache',
            'migrations',
            'scripts/refactor_data_paths.py',  # Don't modify this script
            'scripts/migrate_to_parquet.py'   # Migration script is separate
        ]

        for py_file in python_files:
            # Skip if in excluded directory
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue

            # Check if file contains data-related code
            try:
                content = py_file.read_text()
                if any(keyword in content.lower() for keyword in ['csv', 'data', 'read_csv', 'pandas']):
                    self.target_files.append(py_file)
            except Exception as e:
                logger.warning(f"Couldn't read {py_file}: {e}")

        logger.info(f"Found {len(self.target_files)} Python files to process")

    def _extract_hardcoded_paths(self, content: str) -> List[Tuple[str, str, str]]:
        """
        Extract hardcoded paths from file content.

        Returns:
            List of (original_line, path, timeframe) tuples
        """
        found_paths = []

        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in self.path_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    # Skip if it's already using config
                    if 'config' in line.lower() or 'get_data_path' in line:
                        continue

                    # Try to determine timeframe
                    timeframe = self._guess_timeframe(match)
                    if timeframe:
                        found_paths.append((line.strip(), match, timeframe))

        return found_paths

    def _guess_timeframe(self, path: str) -> Optional[str]:
        """Guess timeframe from path."""
        path_lower = path.lower()

        # Check exact filename matches
        for filename, timeframe in self.csv_mappings.items():
            if filename.lower() in path_lower:
                return timeframe

        # Pattern matching for timeframes
        patterns = {
            r'm1|1min|1minute': '1min',
            r'm3|3min|3minute': '3min',
            r'm5|5min|5minute': '5min',
            r'm15|15min|15minute': '15min'
        }

        for pattern, timeframe in patterns.items():
            if re.search(pattern, path_lower):
                return timeframe

        return None

    def _generate_replacement(self, original_line: str, old_path: str, timeframe: str) -> str:
        """Generate replacement code using data configuration."""
        # Different replacement strategies based on context

        if 'pd.read_csv' in original_line or 'pandas.read_csv' in original_line:
            # Replace pandas read_csv calls
            replacement = original_line.replace(
                f'"{old_path}"',
                f'get_data_path("{timeframe}")'
            ).replace(
                f"'{old_path}'",
                f'get_data_path("{timeframe}")'
            )

        elif 'data_path' in original_line.lower() or 'data_file' in original_line.lower():
            # Replace data path assignments
            replacement = original_line.replace(
                f'"{old_path}"',
                f'get_data_path("{timeframe}")'
            ).replace(
                f"'{old_path}'",
                f'get_data_path("{timeframe}")'
            )

        elif 'DataHandler' in original_line:
            # Handle DataHandler initialization
            replacement = original_line.replace(
                f'"{old_path}"',
                f'get_data_path("{timeframe}")'
            ).replace(
                f"'{old_path}'",
                f'get_data_path("{timeframe}")'
            )

        else:
            # Generic replacement
            replacement = original_line.replace(
                f'"{old_path}"',
                f'get_data_path("{timeframe}")'
            ).replace(
                f"'{old_path}'",
                f'get_data_path("{timeframe}")'
            )

        return replacement

    def _add_import_if_needed(self, content: str) -> str:
        """Add import for get_data_path if not already present."""
        # Check if import already exists
        if 'from config.data_config import' in content or 'get_data_path' in content:
            return content

        # Find the best place to add import
        lines = content.split('\n')
        import_line = "from config.data_config import get_data_path"

        # Find last import line
        last_import_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')) and not line.strip().startswith('#'):
                last_import_idx = i

        if last_import_idx >= 0:
            # Insert after last import
            lines.insert(last_import_idx + 1, import_line)
        else:
            # Insert at beginning after docstring/comments
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith(('#', '"""', "'''")):
                    insert_idx = i
                    break
            lines.insert(insert_idx, import_line)
            lines.insert(insert_idx + 1, "")  # Add blank line

        return '\n'.join(lines)

    def refactor_file(self, file_path: Path) -> Dict[str, any]:
        """Refactor a single file."""
        logger.info(f"Processing {file_path.relative_to(self.project_root)}")

        try:
            original_content = file_path.read_text()
            content = original_content

            # Extract hardcoded paths
            hardcoded_paths = self._extract_hardcoded_paths(content)

            if not hardcoded_paths:
                return {'status': 'no_changes', 'changes': 0}

            changes = []
            modified_content = content

            # Apply replacements
            for original_line, old_path, timeframe in hardcoded_paths:
                new_line = self._generate_replacement(original_line, old_path, timeframe)
                if new_line != original_line:
                    modified_content = modified_content.replace(original_line, new_line)
                    changes.append({
                        'old_path': old_path,
                        'timeframe': timeframe,
                        'old_line': original_line,
                        'new_line': new_line
                    })

            if changes:
                # Add import if needed
                modified_content = self._add_import_if_needed(modified_content)

                # Write back to file
                file_path.write_text(modified_content)

                logger.info(f"✅ Refactored {file_path.name}: {len(changes)} changes")
                return {'status': 'modified', 'changes': len(changes), 'details': changes}
            else:
                return {'status': 'no_changes', 'changes': 0}

        except Exception as e:
            logger.error(f"❌ Failed to refactor {file_path}: {e}")
            return {'status': 'error', 'error': str(e)}

    def refactor_all(self) -> Dict[str, any]:
        """Refactor all target files."""
        logger.info("🚀 Starting data path refactoring...")

        results = {
            'total_files': len(self.target_files),
            'modified_files': 0,
            'total_changes': 0,
            'errors': 0,
            'file_results': {}
        }

        for file_path in self.target_files:
            result = self.refactor_file(file_path)
            results['file_results'][str(file_path.relative_to(self.project_root))] = result

            if result['status'] == 'modified':
                results['modified_files'] += 1
                results['total_changes'] += result['changes']
            elif result['status'] == 'error':
                results['errors'] += 1

        # Generate summary
        logger.info(f"🎉 Refactoring complete!")
        logger.info(f"Files processed: {results['total_files']}")
        logger.info(f"Files modified: {results['modified_files']}")
        logger.info(f"Total changes: {results['total_changes']}")
        logger.info(f"Errors: {results['errors']}")

        return results

    def create_config_init(self) -> None:
        """Create __init__.py file for config module."""
        config_dir = self.project_root / 'config'
        config_dir.mkdir(exist_ok=True)

        init_file = config_dir / '__init__.py'
        if not init_file.exists():
            init_content = '''"""
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
'''
            init_file.write_text(init_content)
            logger.info("✅ Created config/__init__.py")

def main():
    """Main refactoring function."""
    print("🔧 Data Path Refactoring Tool")
    print("=" * 50)

    # Initialize refactor tool
    refactor = DataPathRefactor()

    # Create config module init
    refactor.create_config_init()

    # Show preview of files to be processed
    print(f"\nFiles to process: {len(refactor.target_files)}")
    for file_path in refactor.target_files[:10]:  # Show first 10
        print(f"  - {file_path.relative_to(refactor.project_root)}")
    if len(refactor.target_files) > 10:
        print(f"  ... and {len(refactor.target_files) - 10} more")

    # Auto-proceed for automated execution
    print(f"\nProceeding with automated refactoring...")

    # Run refactoring
    results = refactor.refactor_all()

    # Show detailed results
    print(f"\n📊 Detailed Results:")
    print(f"Success rate: {(results['modified_files'] / results['total_files'] * 100):.1f}%")

    if results['errors'] > 0:
        print(f"\n❌ Errors encountered:")
        for file_path, result in results['file_results'].items():
            if result['status'] == 'error':
                print(f"  {file_path}: {result['error']}")

    # Show sample changes
    modified_files = [
        (path, result) for path, result in results['file_results'].items()
        if result['status'] == 'modified'
    ]

    if modified_files:
        print(f"\n✅ Sample changes:")
        sample_file, sample_result = modified_files[0]
        print(f"File: {sample_file}")
        for change in sample_result.get('details', [])[:2]:  # Show first 2 changes
            print(f"  Old: {change['old_line']}")
            print(f"  New: {change['new_line']}")
            print()

if __name__ == "__main__":
    main()