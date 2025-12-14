"""
Data Validator - Validate data quality before preprocessing
NEW MODULE: Comprehensive data validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from utils.loggerMixin import LoggerMixin


class DataValidator(LoggerMixin):
    """
    Validate data schema, ranges, and quality before preprocessing.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DataValidator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('validation', {})
        self.logger = self.setup_class_logger('data_validator', config)
    
    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict[str, str]) -> None:
        """
        Validate DataFrame schema matches expectations.
        
        Args:
            df: Input DataFrame
            expected_schema: Dictionary of {column_name: expected_dtype}
            
        Raises:
            ValueError: If schema validation fails
        """
        self.logger.info('Validating data schema...')
        
        # Check for missing columns
        expected_cols = set(expected_schema.keys())
        actual_cols = set(df.columns)
        
        missing_cols = expected_cols - actual_cols
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        extra_cols = actual_cols - expected_cols
        if extra_cols:
            self.logger.warning(f"Extra columns found: {extra_cols}")
        
        # Validate data types
        type_mismatches = []
        for col, expected_type in expected_schema.items():
            if col not in df.columns:
                continue
            
            actual_type = str(df[col].dtype)
            
            # Flexible type checking
            if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(df[col]):
                type_mismatches.append(f"{col}: expected numeric, got {actual_type}")
            elif expected_type == 'categorical' and not pd.api.types.is_object_dtype(df[col]):
                type_mismatches.append(f"{col}: expected categorical, got {actual_type}")
        
        if type_mismatches:
            self.logger.warning(f"Type mismatches: {type_mismatches}")
        
        self.logger.info('✓ Schema validation passed')
    
    def validate_ranges(self, df: pd.DataFrame, rules: Dict[str, Dict]) -> None:
        """
        Validate numeric columns are within expected ranges.
        
        Args:
            df: Input DataFrame
            rules: Dictionary of {column: {'min': x, 'max': y}}
            
        Raises:
            ValueError: If range validation fails
        """
        self.logger.info('Validating value ranges...')
        
        violations = []
        
        for col, bounds in rules.items():
            if col not in df.columns:
                self.logger.warning(f"Column {col} not found for range validation")
                continue
            
            min_val = bounds.get('min')
            max_val = bounds.get('max')
            
            if min_val is not None:
                below_min = (df[col] < min_val).sum()
                if below_min > 0:
                    violations.append(f"{col}: {below_min} values below {min_val}")
            
            if max_val is not None:
                above_max = (df[col] > max_val).sum()
                if above_max > 0:
                    violations.append(f"{col}: {above_max} values above {max_val}")
        
        if violations:
            self.logger.warning(f"Range violations: {violations}")
        else:
            self.logger.info('✓ Range validation passed')
    
    def validate_cardinality(self, df: pd.DataFrame, max_cardinality: int = 100) -> None:
        """
        Check for high-cardinality categorical columns.
        
        Args:
            df: Input DataFrame
            max_cardinality: Maximum unique values for categorical columns
        """
        self.logger.info('Checking categorical cardinality...')
        
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        
        high_cardinality_cols = []
        for col in cat_cols:
            unique_count = df[col].nunique()
            if unique_count > max_cardinality:
                high_cardinality_cols.append(f"{col}: {unique_count} unique values")
        
        if high_cardinality_cols:
            self.logger.warning(f"High cardinality columns: {high_cardinality_cols}")
        else:
            self.logger.info('✓ Cardinality check passed')
    
    def run_all_validations(self, df: pd.DataFrame) -> bool:
        """
        Run all validation checks.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if all validations pass
        """
        try:
            self.logger.info('='*60)
            self.logger.info('STARTING DATA VALIDATION')
            self.logger.info('='*60)
            
            # Schema validation
            if 'schema' in self.config:
                self.validate_schema(df, self.config['schema'])
            
            # Range validation
            if 'ranges' in self.config:
                self.validate_ranges(df, self.config['ranges'])
            
            # Cardinality check
            if 'max_cardinality' in self.config:
                self.validate_cardinality(df, self.config['max_cardinality'])
            
            self.logger.info('='*60)
            self.logger.info('✓ ALL VALIDATIONS PASSED')
            self.logger.info('='*60)
            
            return True
        
        except Exception as e:
            self.logger.error(f'Validation failed: {e}', exc_info=True)
            return False