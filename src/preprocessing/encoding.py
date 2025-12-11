
import pandas as pd
import numpy as np
from utils import setup_logger, ensure_directory
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FeatureEncoder:
    '''Encode categorical features'''
    def __init__(self, config):
        self.config = config['encoding']
        self.logger = self._setup_logging()
        self.encoding_cache = {}

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

    def encode_features(self, df, fit=True):
        try:
            self.logger.info(f'Encoding categorical features...')

            # one-hot encoding
            df = self._one_hot_encode(df, fit)

            self.logger.info(f'Feature encoding completed')
            return df
        
        except Exception as e:
            self.logger.error(f'Error encoding features: {e}')
            raise

    def _one_hot_encode(self, df, fit=True):
        '''one-hot encode low cardinality features'''
        try:
            if not self.config['one_hot_columns']:
                self.logger.warning(f'One Hot Encoding disabled')
                return df

            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            for col in categorical_cols:
                    
                dummies = pd.get_dummies(data=df[col], prefix=col, drop_first=True, dtype=int)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])

                self.logger.debug(f'On-hot encoded {col} into {len(dummies.columns)} features')

            return df
        
        except Exception as e:
            self.logger.error(f'Error in one-hot encoding: {e}')
            raise
        