

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split


from utils.loggerMixin import LoggerMixin


# UPDATED: Inherit from LoggerMixin
class DataSplitter(LoggerMixin):
    """
    Split data before transformations to prevent leakage.
    
    FIXED: Split logic (was splitting full df twice)
    UPDATED: Type hints, validation, documentation
    """
    
    def __init__(self, config: dict):
        self.config = config['data_split']

        self.logger = self.setup_class_logger('data_splitter', config, 'logging')
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/dev/test sets with stratification.
        
        ✅ FIXED: Now correctly splits train_dev_set (not df) in second split
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (train_set, dev_set, test_set) with no overlap
        
        Raises:
            ValueError: If split sizes are invalid
        
        Example:
            >>> splitter = DataSplitter(config)
            >>> train, dev, test = splitter.split_data(df)
        """
        self.logger.info('Starting data split...')
        
        try:
            test_size = self.config.get('test_size', 610)
            dev_size = self.config.get('dev_size', 500)
            random_state = self.config['random_state']
            stratify_col = self.config.get('stratify_column', 'stroke')
            
            total_size = len(df)
            self.logger.info(f'Total observations: {total_size}')
            
            # ✨ NEW: Validate split sizes
            if test_size + dev_size >= total_size:
                raise ValueError(
                    f"test_size ({test_size}) + dev_size ({dev_size}) "
                    f"must be less than total ({total_size})"
                )
            
            # First split: separate test set
            stratify = df[stratify_col] if stratify_col in df.columns else None
            
            train_dev_set, test_set = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state,
                stratify=stratify
            )
            
            self.logger.info(f'Test set separated: {len(test_set)} rows')
            
            # ✅ FIXED: Second split uses train_dev_set (not df)
            stratify_train_dev = (
                train_dev_set[stratify_col] 
                if stratify_col in train_dev_set.columns 
                else None
            )
            
            train_set, dev_set = train_test_split(
                train_dev_set,  # ✅ FIXED: Was df, now train_dev_set
                test_size=dev_size, 
                random_state=random_state,
                stratify=stratify_train_dev  # ✅ FIXED: Stratify on train_dev_set
            )
            
            # Log split statistics
            self.logger.info(f'Train set: {len(train_set)} rows ({len(train_set)/total_size*100:.1f}%)')
            self.logger.info(f'Dev set:   {len(dev_set)} rows ({len(dev_set)/total_size*100:.1f}%)')
            self.logger.info(f'Test set:  {len(test_set)} rows ({len(test_set)/total_size*100:.1f}%)')
            
            # ✨ NEW: Validate split
            self._validate_split(train_set, dev_set, test_set, total_size, stratify_col)
            
            return (
                train_set.reset_index(drop=True), 
                dev_set.reset_index(drop=True), 
                test_set.reset_index(drop=True)
            )
        
        except Exception as e:
            self.logger.error(f'Error during data split: {e}', exc_info=True)
            raise
    
    def _validate_split(self, train: pd.DataFrame, dev: pd.DataFrame, 
                       test: pd.DataFrame, original_size: int, 
                       stratify_col: str) -> None:
        """
        ✨ NEW: Enhanced validation with size and distribution checks.
        
        Args:
            train: Training set
            dev: Development set
            test: Test set
            original_size: Original DataFrame size
            stratify_col: Column used for stratification
        
        Raises:
            ValueError: If validation fails
        """
        # Check total rows
        split_total = len(train) + len(dev) + len(test)
        
        if split_total != original_size:
            raise ValueError(
                f'Row count mismatch: {split_total} split vs {original_size} original. '
                f'Data loss detected!'
            )
        
        self.logger.info('✓ No data loss during splitting')
        
        # Check stratification quality
        if stratify_col and stratify_col in train.columns:
            train_dist = train[stratify_col].value_counts(normalize=True)
            dev_dist = dev[stratify_col].value_counts(normalize=True)
            test_dist = test[stratify_col].value_counts(normalize=True)
            
            self.logger.info(f'Target distribution:')
            self.logger.info(f'  Train: {train_dist.to_dict()}')
            self.logger.info(f'  Dev:   {dev_dist.to_dict()}')
            self.logger.info(f'  Test:  {test_dist.to_dict()}')
            
            # Check if distributions are similar (within 5%)
            max_diff_dev = abs(train_dist - dev_dist).max()
            max_diff_test = abs(train_dist - test_dist).max()
            
            if max_diff_dev > 0.05:
                self.logger.warning(
                    f'⚠️  Train-Dev distribution differs by {max_diff_dev:.1%}'
                )
            if max_diff_test > 0.05:
                self.logger.warning(
                    f'⚠️  Train-Test distribution differs by {max_diff_test:.1%}'
                )
            
            if max_diff_dev <= 0.05 and max_diff_test <= 0.05:
                self.logger.info('✓ Stratification maintained across splits')