
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from utils import setup_logger, ensure_directory, write_csv


# ======================
# data quality workflow
# =======================
# check for missing values
# check for duplicates
# check for outliers

class DataQualityError(Exception):
    pass

class DataQuality:
    def __init__(self, config):
        self.config = config
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

    def _check_for_missing_values(self, df):
        '''Analyze missing values with thresholds'''
        warning_threshold = self.config['data_quality'].get('missing_threshold_warning', 0.05)
        critical_threshold = self.config['data_quality'].get('missing_threshold_critical', 0.30)

        self.logger.info(f'Analyzing missing values...')

        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) > 0:
            self.logger.info(f'No missing values found!')

        
        missing_pct = (missing / len(df)).round(4)

        missing_df = pd.DataFrame({
            'missing_count' : missing,
            'missing_percentage' : missing_pct,
            'severity' : ['CRITICAL' if pct >= critical_threshold
                        else 'WARNING' if pct >= warning_threshold
                        else 'INFO'
                        for pct in missing_pct]
        })
        missing_values_artifacts_path = self.config['data'].get('artifact_data_path_missing')

        write_csv(missing_df, missing_values_artifacts_path)

        critical_cols = missing_df[missing_df['severity'] == 'CRITICAL']
        warning_cols = missing_df[missing_df['severity'] == 'WARNING']

        self.logger.info(f'Missing values found in {len(missing_df)} columns')
        if len(critical_cols) > 0:
            self.logger.warning(f'CRITICAL: {len(critical_cols)} columns exceeded {critical_threshold*100}% missing')
        if len(warning_cols) > 0:
            self.logger.warning(f'WARNING: {len(warning_cols)} columns exceeded {warning_threshold*100}% missing')

    def _check_for_duplicates(self, df):
        '''Check for duplicate rows in the dataset'''
        if self.config['data_quality'].get('check_duplicates', True):
            self.logger.info(f'Checking for duplicates in data...')

            duplicates = df.duplicated()
            n_duplicates = df.duplicated().sum()

            if n_duplicates == 0:
                self.logger.info(f'No duplicates found in dataset')

            duplicates_pct = (n_duplicates / len(df)) * 100
            self.logger.warning(f'Found {n_duplicates} duplicates rows ({duplicates_pct:.2f}%)')

        else:
            self.logger.info(f'Duplicates checking disabled! (skipping...)')


    def detect_outliers_iqr(self, df):
        if self.config['data_quality'].get('check_oultiers', True) and self.config['data_quality'].get('outlier_report', True):
            self.logger.info(f'Detecting outliers uisng IQR method')

            cols_not_for_outliers = ['id','stroke','hypertension','heart_disease']

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cols_to_check = [col for col in numeric_cols if col not in cols_not_for_outliers]

            outlier_summary = {}

            for col in cols_to_check:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)

                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                n_outliers = outlier_mask.sum()
                outlier_pct = (n_outliers / len(df)) * 100

                outlier_summary[col] = {
                    'count' : int(n_outliers),
                    'percentage' : round(outlier_pct,2),
                    'lower_bound' : round(lower_bound,2),
                    'upper_bound' : round(upper_bound,2),
                    'Q1' : round(Q1,2),
                    'Q3' : round(Q3,2),
                    'IQR' : round(IQR, 2)
                } 

            summary_df = pd.DataFrame(outlier_summary)
            outlier_summary_path = self.config['data']['artifact_data_path_outliers']

            write_csv(summary_df, outlier_summary_path)

        else:
            self.logger.warning(f'Outlier detection disabled! (skipping...)')