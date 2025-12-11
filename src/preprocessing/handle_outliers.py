
import pandas as pd
import numpy as np
from utils import setup_logger, ensure_directory
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HandleOutliers:
    '''Handler outliers without removing them'''

    def __init__(self, config):
        self.config = config['outliers']
        self.logger = self._setup_logging()
        self.outlier_bounds = {}

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

    def handle_outliers(self, df, fit=True):
        try:
            self.logger.info(f'Processing outliers...')

            cols_to_flag = self.config['cols_to_flag']

            if fit:
                self.logger.info(f'Computing outlier bounds from training data...')
                self.outlier_bounds = self._compute_bounds(df, cols_to_flag)

            df = self._flag_outliers(df, cols_to_flag)

            outlier_count = (df['is_outlier'] == 1).sum()
            self.logger.info(f'Rows flagged as outlier: {outlier_count} ({outlier_count/len(df) * 100:.2f}%)')

            return df
        
        except Exception as e:
            self.logger.error(f'Error handling outliers: {e}')
            raise

    def _compute_bounds(self, df, cols_to_flag):
        '''compute IQR for each column to be flagged'''
        bounds = {}
        
        for col in cols_to_flag:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            bounds[col] = {'lower': lower_bound, 'upper': upper_bound}

        self.logger.debug(f'Completed outlier bounds for {len(cols_to_flag)} columns')
        return bounds
    
    def _flag_outliers(self, df, cols_to_flag):
        '''Flag rows as outliers based on training bounds'''
        is_outlier = pd.Series(0, index=df.index)

        for col in cols_to_flag:
            if col not in self.outlier_bounds:
                continue

            lower = self.outlier_bounds[col]['lower']
            upper = self.outlier_bounds[col]['upper']

            col_outliers = (df[col] < lower) | (df[col] > upper)
            is_outlier = is_outlier | col_outliers.astype(int)

        df['is_outlier'] = is_outlier
        return df