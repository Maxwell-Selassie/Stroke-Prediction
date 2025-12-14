"""
Duplicate Handler - Detect and remove duplicate rows
FIXED: Logic error with duplicates_count.sum()
"""

import pandas as pd
import numpy as np
from pathlib import Path

from utils import ensure_directory, setup_logger
from utils.loggerMixin import LoggerMixin


class DuplicateHandler(LoggerMixin):
    """Handle duplicate rows in dataset."""
    
    def __init__(self, config: dict):
        """
        Initialize DuplicateHandler.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config['duplicates']
        self.full_config = config
        self.logger = self.setup_class_logger('duplicate_handler', config)
    
    def handle_duplicates(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Detect and remove exact duplicates.
        
        FIXED: Was calling duplicates_count.sum() on an integer
        
        Args:
            df: Input DataFrame
            fit: Unused (kept for consistency)
            
        Returns:
            DataFrame with duplicates removed
        """
        try:
            self.logger.info(f'Checking for duplicates - Before: {len(df):,} rows')
            
            if not self.config.get('check_duplicates', True):
                self.logger.info('Duplicate checking disabled (skipping...)')
                return df
            
            duplicates_count = df.duplicated().sum()
            

            if duplicates_count == 0:
                self.logger.info('✓ No duplicates found')
                return df
            
            duplicates_pct = (duplicates_count / len(df)) * 100
            self.logger.warning(
                f'⚠️  Found {duplicates_count} duplicate rows ({duplicates_pct:.2f}%)'
            )
            
            # Remove duplicates
            if self.config.get('remove_duplicates', True):
                df = df.drop_duplicates()
                self.logger.info(f'✓ Duplicates removed - After: {len(df):,} rows')
            
            return df
        
        except Exception as e:
            self.logger.error(f'Error handling duplicates: {e}', exc_info=True)
            raise