"""
Outlier Handler - Flag outliers using IQR method
ENHANCED: Better logging, configurable multiplier
"""

import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from utils import setup_logger, ensure_directory
from utils.loggerMixin import LoggerMixin


class OutlierHandler(LoggerMixin):
    """
    Flag outliers without removing them.
    
    Uses IQR method: outliers are values outside [Q1 - k*IQR, Q3 + k*IQR]
    where k is typically 1.5
    """
    
    def __init__(self, config: dict):
        """
        Initialize OutlierHandler.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config['outliers']
        self.full_config = config
        self.logger = self.setup_class_logger('outlier_handler', config)
        self.outlier_bounds = {}
    
    def handle_outliers(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Flag outliers using IQR method.
        
        Args:
            df: Input DataFrame
            fit: If True, compute bounds; if False, use cached bounds
            
        Returns:
            DataFrame with 'is_outlier' column added
        """
        try:
            self.logger.info('Processing outliers...')
            
            cols_to_flag = self.config.get('cols_to_flag', [])
            
            if not cols_to_flag:
                self.logger.warning('No columns specified for outlier detection')
                df['is_outlier'] = 0
                return df
            
            if fit:
                self.logger.info(f'Computing outlier bounds for {len(cols_to_flag)} columns...')
                self.outlier_bounds = self._compute_bounds(df, cols_to_flag)
            
            df = self._flag_outliers(df, cols_to_flag)
            
            outlier_count = (df['is_outlier'] == 1).sum()
            outlier_pct = (outlier_count / len(df)) * 100
            self.logger.info(
                f'Rows flagged as outliers: {outlier_count:,} ({outlier_pct:.2f}%)'
            )
            
            return df
        
        except Exception as e:
            self.logger.error(f'Error handling outliers: {e}', exc_info=True)
            raise
    
    def _compute_bounds(self, df: pd.DataFrame, cols_to_flag: list) -> Dict[str, Dict[str, float]]:
        """
        Compute IQR bounds for each column.
        
        Args:
            df: Input DataFrame
            cols_to_flag: List of columns to compute bounds for
            
        Returns:
            Dictionary of {column: {'lower': float, 'upper': float}}
        """
        bounds = {}
        multiplier = self.config.get('multiplier', 1.5)
        
        for col in cols_to_flag:
            if col not in df.columns:
                self.logger.warning(f'Column {col} not found for outlier detection')
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            bounds[col] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }
            
            self.logger.debug(
                f'{col}: bounds=[{lower_bound:.2f}, {upper_bound:.2f}], IQR={IQR:.2f}'
            )
        
        self.logger.info(f'✓ Computed bounds for {len(bounds)} columns')
        return bounds
    
    def _flag_outliers(self, df: pd.DataFrame, cols_to_flag: list) -> pd.DataFrame:
        """
        Flag rows as outliers based on computed bounds.
        
        Args:
            df: Input DataFrame
            cols_to_flag: List of columns to check
            
        Returns:
            DataFrame with 'is_outlier' column
        """
        is_outlier = pd.Series(0, index=df.index)
        
        for col in cols_to_flag:
            if col not in self.outlier_bounds:
                self.logger.warning(f'No bounds cached for {col}')
                continue
            
            if col not in df.columns:
                continue
            
            lower = self.outlier_bounds[col]['lower']
            upper = self.outlier_bounds[col]['upper']
            
            col_outliers = (df[col] < lower) | (df[col] > upper)
            is_outlier = is_outlier | col_outliers.astype(int)
            
            n_outliers = col_outliers.sum()
            if n_outliers > 0:
                self.logger.debug(f'{col}: {n_outliers} outliers detected')
        
        df['is_outlier'] = is_outlier
        return df