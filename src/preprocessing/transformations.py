
import pandas as pd
import numpy as np
from utils import setup_logger, ensure_directory
from pathlib import Path

class FeatureTransformer:
    '''Apply mathematical transformations'''

    def __init__(self, config):
        self.config = config['transformations']
        self.logger = self._setup_logging()

    def _setup_logging(self):
        '''setup logging system'''
        try:
            log_config = self.config.get('logging',{})
            log_dir = Path(log_config.get('log_dir','logs/'))

            ensure_directory(log_dir)

            logger = setup_logger(
                name='data_quality',
                log_dir=log_dir,
                log_level=log_config.get('log_level', 'INFO'),
                max_bytes=log_config.get('max_bytes', 10485760),
                backup_count=log_config.get('backup_count', 7)
            )
            return logger 
        except Exception as e:
            print(f'Error setting up logging system: {e}')
        
    def transform_features(self, df, fit=True):
        '''Apply log transformations to reduce skewness'''
        try:
            self.logger.info('Applying feature transformations...')

            log_cols = self.config['log_columns']

            for col in log_cols:
                if col not in df.columns:
                    self.logger.warning(f'Column {col} not found for log transformation')
                    continue

                df[f'{col}_log'] = np.log1p(df[col])
                self.logger.debug(f'Log transformed {col}')
            
            self.logger.info('Feature transformations completed')
            return df
        
        except Exception as e:
            self.logger.error(f'Error in feature transformation: {e}')
            raise