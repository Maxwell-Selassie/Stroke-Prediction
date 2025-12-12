
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from utils import setup_logger, ensure_directory
from sklearn.model_selection import train_test_split

class DataSplitter:
    '''split data before transformations to prevent leakages'''
    def __init__(self, config):
        self.config = config['data_split']
        self.logger = self._setup_logging()

    def _setup_logging(self):
        '''setup logging system'''
        try:
            log_config = self.config.get('logging',{})
            log_dir = Path(log_config.get('log_dir','logs/'))

            ensure_directory(log_dir)

            logger = setup_logger(
                name='split_data',
                log_dir=log_dir,
                log_level=log_config.get('log_level', 'INFO'),
                max_bytes=log_config.get('max_bytes', 10485760),
                backup_count=log_config.get('backup_count', 7)
            )
            return logger 
        except Exception as e:
            print(f'Error setting up logging system: {e}')

    def split_data(self, df):
        '''split data into train/dev/test sets'''
        self.logger.info(f'Starting data split...')

        try:
            test_size = self.config.get('test_size', 610)
            dev_size = self.config.get('dev_size', 500)
            random_state = self.config['random_state']
            stratify_col = self.config.get('stratify_column','stroke')

            total_size = len(df)
            self.logger.info(f'Total observations: {total_size}')

            # first split: separate test set
            train_dev_set, test_set = train_test_split(
                df, test_size=test_size, random_state=random_state,
                stratify=df[stratify_col] if stratify_col else None
            )

            # second split: separate dev from train
            train_set, dev_set = train_test_split(
                df, test_size=dev_size, random_state=random_state,
                stratify=df[stratify_col] if stratify_col else None
            )

            self.logger.info(f'Train set: {len(train_set)} rows ({(len(train_set)/ total_size)*100:.1f}%)')
            self.logger.info(f'Dev set: {len(dev_set)} rows ({len(dev_set) / total_size*100:.1f}%)')
            self.logger.info(f"Test set: {len(test_set)} rows ({len(test_set)/total_size*100:.1f}%)")

            # validate class distributions
            self._validate_split(df, train_set, dev_set, test_set, stratify_col)

            return train_set.reset_index(drop=True), dev_set.reset_index(drop=True), test_set.reset_index(drop=True), stratify_col
        
        except Exception as e:
            self.logger.error(f'Error during data split: {e}')
            raise

    def _validate_split(self, df, train, dev, test, stratify_col):
        '''validate class distribution accross splits'''
        if stratify_col:
            full_dist = df[stratify_col].value_counts(normalize=True)
            train_dist = train[stratify_col].value_counts(normalize=True)
            dev_dist = dev[stratify_col].value_counts(normalize=True)
            test_dist = test[stratify_col].value_counts(normalize=True)
            
            self.logger.info(f"Full dataset {stratify_col} distribution: {full_dist.to_dict()}")
            self.logger.info(f"Train {stratify_col} distribution: {train_dist.to_dict()}")
            self.logger.info(f"Dev {stratify_col} distribution: {dev_dist.to_dict()}")
            self.logger.info(f"Test {stratify_col} distribution: {test_dist.to_dict()}")


