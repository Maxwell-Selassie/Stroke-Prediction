
import pandas as pd
import numpy as np
from utils import ensure_directory, setup_logger
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

class DuplicateHandler:
    'Handle duplicate rows'

    def __init__(self, config):
        self.config = config['duplicates']
        self.logger = self._setup_logging()

    def _setup_logging(self):
        '''setup logging system'''
        try:
            log_config = self.config.get('logging',{})
            log_dir = Path(log_config.get('log_dir','logs/'))

            ensure_directory(log_dir)

            logger = setup_logger(
                name='handle_duplicates',
                log_dir=log_dir,
                log_level=log_config.get('log_level', 'INFO'),
                max_bytes=log_config.get('max_bytes', 10485760),
                backup_count=log_config.get('backup_count', 7)
            )
            return logger 
        except Exception as e:
            print(f'Error setting up logging system: {e}')

    def handle_duplicates(self, df, fit=True):
        '''Detect and remove exact duplicates'''
        try:
            self.logger.info(f'Checking for duplicates -  Before: {len(df)} rows')

            if self.config['check_duplicates']:
                duplicates_count = df.duplicated().sum()
                self.logger.info(f'Exact duplicates found: {duplicates_count}')

                if duplicates_count.sum() > 0:
                    df = df.drop_duplicates()
                    self.logger.info(f'Duplicated removed - After: {len(df)}')
                else:
                    self.logger.info(f'No duplicates found')

            return df
        
        except Exception as e:
            self.logger.error(f'Error handling duplicates: {e}')
            raise