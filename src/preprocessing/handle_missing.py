
import pandas as pd
import numpy as np
from utils import ensure_directory, setup_logger
from pathlib import Path

class MissingHandler:
    '''Handling missing values'''

    def __init__(self,config):
        self.config = config['missing_values']
        self.logger = self._setup_logging()

    def _setup_logging(self):
        '''setup logging system'''
        try:
            log_config = self.config.get('logging',{})
            log_dir = Path(log_config.get('log_dir','logs/'))

            ensure_directory(log_dir)

            logger = setup_logger(
                name='handle_missing',
                log_dir=log_dir,
                log_level=log_config.get('log_level', 'INFO'),
                max_bytes=log_config.get('max_bytes', 10485760),
                backup_count=log_config.get('backup_count', 7)
            )
            return logger 
        except Exception as e:
            print(f'Error setting up logging system: {e}')

    def handle_missing(self, df, fit=True):
        self.logger.info(f'Handling missing values - Before: {len(df)} rows')
        
        if not self.config['enabled']:
            self.logger.warning('Missing value handling disabled')
            return df
        
        # Separate numeric and categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Numeric missing values
        numeric_missing = [col for col in numeric_cols if df[col].isnull().any()]
        if numeric_missing:
            numeric_strategy = self.config['numeric']['strategy']
            
            if fit:
                # Compute and cache statistics on train
                if numeric_strategy == 'mean':
                    self.impute_values = {col: df[col].mean() for col in numeric_missing}
                elif numeric_strategy == 'median':
                    self.impute_values = {col: df[col].median() for col in numeric_missing}
            
            # Apply imputation
            for col in numeric_missing:
                if col in self.impute_values:
                    df[col].fillna(self.impute_values[col], inplace=True)
                    self.logger.debug(f'Imputed {col} with {numeric_strategy}')
        
        # Categorical missing values
        categorical_missing = [col for col in categorical_cols if df[col].isnull().any()]
        if categorical_missing:
            categorical_strategy = self.config['categorical']['strategy']
            
            if fit:
                # Compute and cache mode on train
                if categorical_strategy == 'mode':
                    self.impute_values_cat = {
                        col: df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                        for col in categorical_missing
                    }
            
            # Apply imputation
            for col in categorical_missing:
                if col in self.impute_values_cat:
                    df[col].fillna(self.impute_values_cat[col], inplace=True)
                    self.logger.debug(f'Imputed {col} with mode')
        
        return df

