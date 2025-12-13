
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import (
    DataSplitter, HandleOutliers, MissingHandler,DuplicateHandler, 
    FeatureTransformer, FeatureEncoder
)
from utils import ensure_directory, setup_logger, Timer, read_yaml, write_csv,read_csv

class PreprocessingPipeline:
    '''Orchestrate all preprocessing steps'''

    def __init__(self):
        self.logger = setup_logger(
            name='preprocessing_pipeline',
            log_dir='logs/'
        )
        self.config = self.load_config()

        self.splitter = DataSplitter(self.config)
        self.missing_handler = MissingHandler(self.config)
        self.duplicate_handler = DuplicateHandler(self.config)
        self.outlier_handler = HandleOutliers(self.config)
        self.feature_encoder = FeatureEncoder(self.config)
        self.feature_transformer = FeatureTransformer(self.config)




    def load_config(self, config_path='config/preprocessing_config.yaml'):
        """Load preprocessing configuration"""
        try:
            with open(config_path, 'r') as f:
                config = read_yaml(config_path)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise

    def _load_data(self):
        '''Load data from file path'''
        try:
            file_path = Path(self.config.get('file_path','data/raw/healthcare-dataset-stroke-data.csv'))
            if not file_path.exists():
                raise FileNotFoundError(f'ERROR: File not found: {file_path}')

            df = read_csv(file_path, optimize_dtypes=True)
            self.logger.info(f'DataFrame successfully loaded with shape: {df.shape}')
            return df
        except Exception as e:
            self.logger.error(f'ERROR loading CSV file: {e}')
            raise
    
    def drop_columns(self, df):
        '''Drop 'id' column in dataset'''
        cols_to_drop = self.config['columns_to_drop']
        # existing_cols = [col for col in cols_to_drop if col in df.columns]

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.logger.debug(f'Dropped columns: {cols_to_drop}')

        return df

    def fit_transform(self):
        '''Fit and transform training data, then transform dev/test'''
        try:
            self.logger.info('='*50)
            self.logger.info(f'STARTING PREPROCESSING PIPELINE')
            self.logger.info('='*50)

            # stage 0
            df = self._load_data()

            # stage 1: pre-split data cleaning
            self.logger.info(f'Pre-split data cleaning...')
            df = self.duplicate_handler.handle_duplicates(df)
            df = self.missing_handler.handle_missing(df)
            df = self.feature_encoder.encode_features(df)
            df = self.drop_columns(df)

            # stage 2: split data
            self.logger.info(f'Splitting Data...')
            train_set, dev_set, test_set = self.splitter.split_data(df)

            # stage 3: fit transformers on training set
            self.logger.info(f'Fitting transformers on training data...')
            train_set = self.outlier_handler.handle_outliers(train_set, fit=True)
            # train_set = self.feature_encoder.encode_features(train_set, fit=True)
            train_set = self.feature_transformer.transform_features(train_set, fit=True)
            

            # stage 4: transform dev set
            self.logger.info(f'Transforming dev set')
            dev_set = self.outlier_handler.handle_outliers(dev_set, fit=False)
            # dev_set = self.feature_encoder.encode_features(dev_set, fit=False)
            dev_set = self.feature_transformer.transform_features(dev_set, fit=False)


            # stage 5: transform test set
            self.logger.info(f'Transforming test set')
            test_set = self.outlier_handler.handle_outliers(test_set, fit=False)
            # test_set = self.feature_encoder.encode_features(test_set, fit=False)
            test_set = self.feature_transformer.transform_features(test_set, fit=False)


            self.logger.info(f'Saving outputs...')
            self._save_datasets(train_set, dev_set, test_set)
            self._save_pipeline()

            self.logger.info('='*50)
            self.logger.info(f'PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY')
            self.logger.info('='*50)

            return train_set, dev_set, test_set
        
        except Exception as e:
            self.logger.error(f'Pipeline failed: {e}')
            raise

    def _save_datasets(self, train, dev, test):
        """Save preprocessed datasets"""
        try:
            splits_config = self.config['output']['processed_dir']
            
            write_csv(train, Path(f'{splits_config}/train_set.csv'))
            write_csv(dev, Path(f'{splits_config}/dev_set.csv'))
            write_csv(test, Path(f'{splits_config}/test_set.csv'))
            
            self.logger.info("All datasets saved successfully")
        
        except Exception as e:
            self.logger.error(f"Error saving datasets: {e}")

    def _save_pipeline(self):
        '''serialize pipeline to joblib file for inference'''
        try:
            pipeline_path = self.config['output']['pipeline_file']

            pipeline_obj = {
                'outlier_handler' : self.outlier_handler,
                'encoder' : self.feature_encoder,
                'transformer' : self.feature_transformer,
                'config' : self.config
            }

            joblib.dump(pipeline_obj, pipeline_path)
            self.logger.info(f'Pipeline saved to {pipeline_path}')

        except Exception as e:
            self.logger.error(f'Error saving  pipeline: {e}')
            raise

def main():
    '''Main execution'''

    try:
        pipeline = PreprocessingPipeline()

        # fit - transform
        train_set, dev_set, test_set = pipeline.fit_transform()

        # save datasets
        pipeline._save_datasets(train_set, dev_set, test_set)

        # save pipeline
        pipeline._save_pipeline()


        print(f'Preprocessing completed successfully')

        return train_set, dev_set, test_set

    except Exception as e:
        print(f'Pipeline execution failed: {e}')
        raise

if __name__ == '__main__':
    train, dev, test = main()