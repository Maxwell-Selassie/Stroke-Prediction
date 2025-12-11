
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from eda import DataQuality, DataOverview, Visualizations
from utils import ensure_directory, setup_logger, Timer

class EDAPipelineError(Exception):
    pass

class EDAPipeline:
    def __init__(self, config_path):
        self.config = config_path
        self.logger = setup_logger(
            name='eda_pipeline',
            log_dir=Path('logs/')
        )
        self.logger.info(f'='*50)
        self.logger.info(f'EDA PIPELINE INITIALIZED')
        self.logger.info('='*50)



    def execute(self):
        '''Execute EDA pipeline'''

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

            with Timer('Data visualizations'):
                visuals = Visualizations(config)
                visuals.run_visualizations(df)

            self.logger.info("=" * 80)
            self.logger.info("EDA PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f'FAILED TO RUN EDA PIPELINE: {e}', exc_info=True)
            raise

if __name__ == '__main__':
    pipeline = EDAPipeline(config_path='config/eda_config.yaml')
    pipeline.execute()