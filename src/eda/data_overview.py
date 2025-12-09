'''
Docstring for eda.data_overview
'''

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io_utils import read_yaml, ensure_directory, read_csv, write_json
from utils.logger import setup_logger

class DataExecutionError(Exception):
    pass

class DataOverview:
    def __init__(self, config_path: Union[Path, str]):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

    def _load_config(self, config_path):
        '''Load configuration from YAML file'''
        try:
            config = read_yaml(config_path)
            return config
        except FileNotFoundError:
            print(f'ERROR: config file not found: {config_path}')
        except Exception as e:
            print(f'Failed to load config: {e}')
            sys.exit(1)

    def _setup_logging(self):
        '''setup logging system'''
        try:
            log_config = self.config.get('logging',{})
            log_dir = Path(log_config.get('log_dir','logs/'))

            ensure_directory(log_dir)

            logger = setup_logger(
                name='data_overview',
                log_dir=log_dir,
                log_level=log_config.get('log_level', 'INFO'),
                max_bytes=log_config.get('max_bytes', 10485760),
                backup_count=log_config.get('backup_count', 7)
            )
            return logger 
        except Exception as e:
            print(f'Error setting up logging system: {e}')

    def _load_data(self):
        '''Load data from file path'''
        try:
            file_path = Path(self.config.get('raw_path','data/raw/healthcare-dataset-stroke-data.csv'))
            if not file_path.exists():
                raise FileNotFoundError(f'ERROR: File not found: {file_path}')

            df = read_csv(file_path, optimize_dtypes=True)
            self.logger.info(f'DataFrame successfully loaded with shape: {df.shape}')
            return df
        except Exception as e:
            self.logger.error(f'ERROR loading CSV file: {e}')
            raise

    def _validate_data(self, df: pd.DataFrame):
        '''validate dataframe'''
        if len(df) == 0:
            self.logger.error(f'DataFrame is Empty')
            raise
        expected_columns = self.config['data']['expected_columns']
        try:
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if len(missing_cols) > 0:
                self.logger.error(f'Missing expected features: {missing_cols}')

            unexpected_cols = [col for col in df.columns if not col in expected_columns]
            if len(unexpected_cols) > 0:
                self.logger.error(f'Unexpected features detected: {unexpected_cols}')

            if self.config['data']['save_expected_columns']:
                feature_names_path = self.config['data']['feature_store_path']
                write_json(expected_columns, feature_names_path, indent=4)
                
        except ValueError as e:
            self.logger.error(f'Error validating dataframe: {e}')
            raise
        except Exception as e:
            self.logger.error(f'ERROR validating dataframe: {e}')
            raise

    def _convert_dtypes(self, df: pd.DataFrame):
        '''convert misrepresented features to appropriate data types'''
        try:
            if self.config['data']['dtype_conversions']:
                convert_to_int = self.config['data']['dtype_conversions'].get('convert_to_int',[])
                for col in convert_to_int:
                    df[col] = df[col].astype(int)
                    self.logger.info(f'Succesfully converted {col} to an integer type')
        except ValueError as e:
            self.logger.error(f'FAILED to convert data types: {e}')
            raise
        except Exception as e:
            self.logger.error(f'ERROR converting data types: {e}')
            raise
    
    def _analyze_numeric_cols(self, df: pd.DataFrame):
        '''Analyze numerical columns'''
        self.logger.info(f'Analyzing numerical columns...')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        id_column = self.config['data']['id_column']

        if id_column in numeric_cols:
            numeric_cols.remove(id_column)

        if not numeric_cols:
            self.logger.warning(f'No numerical columns found: {numeric_cols}')
            return pd.DataFrame(),[]
        
        num_summary = df.describe().T
        num_summary['range'] = num_summary['max'] - num_summary['min']

        self.logger.info(f'Finished Analyzing {len(numeric_cols)} numeric : {numeric_cols}')

        return num_summary, numeric_cols
    
    def _analyze_categorical_cols(self, df: pd.DataFrame):
        '''Analyze categorical columns'''
        self.logger.info('Analyzing categorical columns...')

        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if not cat_cols:
            self.logger.warning(f'No categorical columns found: {cat_cols}')
            return pd.DataFrame(), []
        
        cat_summary = df.describe(exclude=[np.number])
        self.logger.info(f'Finished analyzing {len(cat_cols)} categorical columns')
        return cat_summary, cat_cols

    
    def get_config(self):
        return self.config
    
    def run_data_overview(self):
        '''Execute all data overview functions'''
        results = {}

        self.logger.info(f'Starting data overview...')
        results['load_data'] = self._load_data()
        df = results['load_data']
        results['validate_data'] = self._validate_data(df)
        results['convert_dtypes'] = self._convert_dtypes(df)
        results['analyze_numeric_cols'] = self._analyze_numeric_cols(df)
        results['analyze_categorical_cols'] = self._analyze_categorical_cols(df)

        self.logger.info(f'Data Overview done...')
        return results
    
if __name__ == '__main__':
    CONFIG_PATH = Path('config/eda_config.yaml')
    data_overview = DataOverview(CONFIG_PATH)
    data_overview.run_data_overview()
