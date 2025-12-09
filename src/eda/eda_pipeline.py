
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from eda import DataQuality, DataOverview
from utils import ensure_directory, setup_logger, Timer

class EDAPipelineError(Exception):
    pass

class EDAPipeline:
    def __init__(self, config_path):
        self.config = config_path
        self.logger = self._setup_logging()


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




    def execute(self):
        '''Execute EDA pipeline'''
        # self.logger.info(f'='*50)
        # self.logger.info(f'EDA PIPELINE INITIALIZED')
        # self.logger.info('='*50)
        try:
            with Timer('Data overview'):
                overview = DataOverview(self.config)
                results = overview.run_data_overview()

                df = results['load_data']
                config = overview.get_config()

            with Timer('Data Quality Checks'):
                quality_checks = DataQuality(config)
                quality_checks._check_for_missing_values(df)
                quality_checks._check_for_duplicates(df)
                quality_checks.detect_outliers_iqr(df)

            # self.logger.info("=" * 80)
            # self.logger.info("EDA PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            # self.logger.info("=" * 80)

        except Exception as e:
            # self.logger.error(f'FAILED TO RUN EDA PIPELINE: {e}', exc_info=True)
            raise

if __name__ == '__main__':
    pipeline = EDAPipeline(config_path='config/eda_config.yaml')
    pipeline.execute()