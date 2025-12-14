"""
Missing Value Handler - Impute or drop missing values
FIXED: Was applying mean to all columns (including categorical)
ENHANCED: Separate numeric/categorical handling, fit/transform pattern
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from utils import ensure_directory, setup_logger
from utils.loggerMixin import LoggerMixin


class MissingHandler(LoggerMixin):
    """
    Handle missing values with fit/transform pattern.
    
    Supports:
    - Mean/median imputation for numeric columns
    - Mode imputation for categorical columns
    - Forward/backward fill for time series
    """
    
    def __init__(self, config: dict):
        """
        Initialize MissingHandler.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config['missing_values']
        self.full_config = config
        self.logger = self.setup_class_logger('missing_handler', config)
        
        # ENHANCED: Cache imputation values from training set
        self.impute_values_numeric = {}
        self.impute_values_categorical = {}
    
    def handle_missing(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values with appropriate strategies.
        
        FIXED: Was applying mean to all columns (broke on categorical)
        ENHANCED: Separate numeric and categorical handling
        
        Args:
            df: Input DataFrame
            fit: If True, compute imputation values; if False, use cached values
            
        Returns:
            DataFrame with missing values handled
            
        Raises:
            ValueError: If imputation fails
        """
        try:
            self.logger.info(f'Handling missing values - Before: {len(df):,} rows')
            
            if not self.config.get('enabled', True):
                self.logger.warning('Missing value handling disabled (skipping...)')
                return df
            
            # Get columns with missing values
            missing_cols = df.columns[df.isnull().any()].tolist()
            
            if not missing_cols:
                self.logger.info('✓ No missing values found')
                return df
            
            self.logger.info(f'Found {len(missing_cols)} columns with missing values')
            
            # FIXED: Separate numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            
            # Handle numeric missing values
            numeric_missing = [col for col in missing_cols if col in numeric_cols]
            if numeric_missing:
                df = self._handle_numeric_missing(df, numeric_missing, fit)
            
            # Handle categorical missing values
            categorical_missing = [col for col in missing_cols if col in categorical_cols]
            if categorical_missing:
                df = self._handle_categorical_missing(df, categorical_missing, fit)
            
            # Check remaining missing values
            remaining_missing = df.isnull().sum().sum()
            if remaining_missing > 0:
                self.logger.warning(f'⚠️  {remaining_missing} missing values remain')
            else:
                self.logger.info('✓ All missing values handled')
            
            self.logger.info(f'Handling missing values - After: {len(df):,} rows')
            
            return df
        
        except Exception as e:
            self.logger.error(f'Error handling missing values: {e}', exc_info=True)
            raise ValueError(f'Missing value handling failed: {e}')
    
    def _handle_numeric_missing(
        self,
        df: pd.DataFrame,
        missing_cols: list,
        fit: bool
    ) -> pd.DataFrame:
        """
        Handle missing values in numeric columns.
        
        Args:
            df: Input DataFrame
            missing_cols: List of numeric columns with missing values
            fit: If True, compute statistics; if False, use cached
            
        Returns:
            DataFrame with numeric missing values handled
        """
        numeric_strategy = self.config['numeric'].get('strategy', 'mean')
        
        self.logger.info(f'Handling {len(missing_cols)} numeric columns with {numeric_strategy} strategy')
        
        if fit:
            # Compute and cache statistics on training set
            for col in missing_cols:
                if numeric_strategy == 'mean':
                    self.impute_values_numeric[col] = df[col].mean()
                elif numeric_strategy == 'median':
                    self.impute_values_numeric[col] = df[col].median()
                elif numeric_strategy == 'zero':
                    self.impute_values_numeric[col] = 0
                else:
                    self.logger.warning(f'Unknown strategy {numeric_strategy}, using mean')
                    self.impute_values_numeric[col] = df[col].mean()
        
        # Apply imputation
        for col in missing_cols:
            if col in self.impute_values_numeric:
                impute_value = self.impute_values_numeric[col]
                df[col].fillna(impute_value, inplace=True)
                self.logger.debug(f'✓ Imputed {col} with {numeric_strategy}={impute_value:.2f}')
            else:
                self.logger.warning(f'No cached value for {col}, using column mean')
                df[col].fillna(df[col].mean(), inplace=True)
        
        return df
    
    def _handle_categorical_missing(
        self,
        df: pd.DataFrame,
        missing_cols: list,
        fit: bool
    ) -> pd.DataFrame:
        """
        Handle missing values in categorical columns.
        
        Args:
            df: Input DataFrame
            missing_cols: List of categorical columns with missing values
            fit: If True, compute mode; if False, use cached
            
        Returns:
            DataFrame with categorical missing values handled
        """
        categorical_strategy = self.config['categorical'].get('strategy', 'mode')
        
        self.logger.info(f'Handling {len(missing_cols)} categorical columns with {categorical_strategy} strategy')
        
        if fit:
            # Compute and cache mode on training set
            for col in missing_cols:
                if categorical_strategy == 'mode':
                    mode_values = df[col].mode()
                    self.impute_values_categorical[col] = mode_values[0] if len(mode_values) > 0 else 'Unknown'
                elif categorical_strategy == 'constant':
                    self.impute_values_categorical[col] = self.config['categorical'].get('fill_value', 'Unknown')
                else:
                    self.logger.warning(f'Unknown strategy {categorical_strategy}, using mode')
                    mode_values = df[col].mode()
                    self.impute_values_categorical[col] = mode_values[0] if len(mode_values) > 0 else 'Unknown'
        
        # Apply imputation
        for col in missing_cols:
            if col in self.impute_values_categorical:
                impute_value = self.impute_values_categorical[col]
                df[col].fillna(impute_value, inplace=True)
                self.logger.debug(f'✓ Imputed {col} with {categorical_strategy}={impute_value}')
            else:
                self.logger.warning(f'No cached value for {col}, using Unknown')
                df[col].fillna('Unknown', inplace=True)
        
        return df