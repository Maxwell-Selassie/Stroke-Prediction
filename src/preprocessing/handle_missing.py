
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
        try:
            self.logger.info(f'Handling missing_values - Before: {len(df)} rows')


            missing_cols = df.columns[df.isnull().any()].tolist()

            if not self.config['enabled']:
                self.logger.warning(f'Handling missing values is not enabled (skipping...)')
                return None
            
            numeric_strategy = self.config['numeric']['strategy']
            if numeric_strategy == 'mean':
                for col in missing_cols:
                    df[col].fillna(df[col].mean(), inplace=True)

            # if categorical_cols:
            #     categorical_strategy = self.config['categorical']['strategy']
            #     if categorical_strategy == 'mode':
            #         cat_missing_cols = df[missing_cols].select_dtypes(exclude=[np.number]).columns.tolist()
            #         if cat_missing_cols:
            #             for col in cat_missing_cols:
            #                 df = df[col].fillna(df[col].mode())

            missing_summary = df.isnull().sum()
            if missing_summary.sum() > 0:
                self.logger.warning(f'Remaining missing values: \n{missing_summary[missing_summary > 0]}')
            else:
                self.logger.info(f'No missing values remaining')

            return df
    
        except Exception as e:
            self.logger.error(f'Error handling missing values: {e}')
            raise


