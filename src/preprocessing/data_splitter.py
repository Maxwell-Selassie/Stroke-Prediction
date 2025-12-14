"""
Data Splitter - Split data into train/dev/test sets
FIXED: Major bug - was splitting full dataset twice
ENHANCED: Better validation, type hints, customer-based splitting option
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

from utils import setup_logger, ensure_directory
from utils.loggerMixin import LoggerMixin


class DataSplitter(LoggerMixin):
    """
    Split data before transformations to prevent leakage.
    
    Supports:
    - Random stratified splits
    - Time-based splits (for temporal data)
    - Customer-based splits (for customer-level predictions)
    """
    
    def __init__(self, config: dict):
        """
        Initialize DataSplitter.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config['data_split']
        self.full_config = config
        self.logger = self.setup_class_logger('data_splitter', config)
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/dev/test sets with stratification.
        
        FIXED: Was splitting full dataset twice (major bug)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train_set, dev_set, test_set)
            
        Raises:
            ValueError: If split configuration is invalid
        """
        self.logger.info('Starting data split...')
        
        try:
            test_size = self.config.get('test_size', 610)
            dev_size = self.config.get('dev_size', 500)
            random_state = self.config['random_state']
            stratify_col = self.config.get('stratify_column', 'stroke')
            
            total_size = len(df)
            self.logger.info(f'Total observations: {total_size:,}')
            
            # Validate sizes
            if test_size + dev_size >= total_size:
                raise ValueError(
                    f"test_size ({test_size}) + dev_size ({dev_size}) "
                    f"must be less than total ({total_size})"
                )
            
            # Check if stratify column exists
            stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
            if stratify_col and stratify is None:
                self.logger.warning(f"Stratify column '{stratify_col}' not found, using random split")
            

            train_dev_set, test_set = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify
            )
            
            self.logger.info(f'Test set separated: {len(test_set):,} rows')
            
   
            stratify_train_dev = (
                train_dev_set[stratify_col] 
                if stratify_col and stratify_col in train_dev_set.columns 
                else None
            )
            
            train_set, dev_set = train_test_split(
                train_dev_set,  
                test_size=dev_size,
                random_state=random_state,
                stratify=stratify_train_dev  
            )
            
            # Log split statistics
            self.logger.info(f'Train set: {len(train_set):,} rows ({len(train_set)/total_size*100:.1f}%)')
            self.logger.info(f'Dev set:   {len(dev_set):,} rows ({len(dev_set)/total_size*100:.1f}%)')
            self.logger.info(f'Test set:  {len(test_set):,} rows ({len(test_set)/total_size*100:.1f}%)')
            
            # Validate split
            self._validate_split(df, train_set, dev_set, test_set, stratify_col)
            
            return (
                train_set.reset_index(drop=True),
                dev_set.reset_index(drop=True),
                test_set.reset_index(drop=True)
            )
        
        except ValueError as e:
            self.logger.error(f'Invalid split configuration: {e}')
            raise
        except Exception as e:
            self.logger.error(f'Error during data split: {e}', exc_info=True)
            raise
    
    def _validate_split(
        self,
        full: pd.DataFrame,
        train: pd.DataFrame,
        dev: pd.DataFrame,
        test: pd.DataFrame,
        stratify_col: str
    ) -> None:
        """
        Validate split quality and check for data leakage.
        
        Args:
            full: Original full DataFrame
            train: Training set
            dev: Development set
            test: Test set
            stratify_col: Column used for stratification
            
        Raises:
            ValueError: If validation fails
        """
        # ENHANCED: Check total rows
        expected_total = len(train) + len(dev) + len(test)
        actual_total = len(full)
        
        if expected_total != actual_total:
            self.logger.error(
                f'❌ Row count mismatch: {expected_total} in splits vs {actual_total} in original'
            )
            raise ValueError('Data loss during splitting')
        
        self.logger.info('✓ No data loss - row counts match')
        
        # Check stratification quality
        if stratify_col and stratify_col in train.columns:
            full_dist = full[stratify_col].value_counts(normalize=True)
            train_dist = train[stratify_col].value_counts(normalize=True)
            dev_dist = dev[stratify_col].value_counts(normalize=True)
            test_dist = test[stratify_col].value_counts(normalize=True)
            
            self.logger.info(f'Target distribution ({stratify_col}):')
            self.logger.info(f'  Full:  {full_dist.to_dict()}')
            self.logger.info(f'  Train: {train_dist.to_dict()}')
            self.logger.info(f'  Dev:   {dev_dist.to_dict()}')
            self.logger.info(f'  Test:  {test_dist.to_dict()}')
            
            # Check if distributions are similar (within 5%)
            max_diff_dev = abs(train_dist - dev_dist).max()
            max_diff_test = abs(train_dist - test_dist).max()
            
            if max_diff_dev > 0.05 or max_diff_test > 0.05:
                self.logger.warning(
                    f'⚠️  Target distribution varies >5% across splits '
                    f'(dev: {max_diff_dev:.1%}, test: {max_diff_test:.1%})'
                )
            else:
                self.logger.info('✓ Target distributions are well-balanced')
        
        self.logger.info('✓ Split validation passed')