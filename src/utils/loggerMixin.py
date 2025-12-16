
from pathlib import Path
from typing import Optional
from utils import setup_logger, ensure_directory

class LoggerMixin:
    ''' 
    Mixin to add standardized logging capabilities to any class

    Usage: 
        class MyClass(LoggerMixin):
        def __init__(self):
        selff.logger = self.setup_class_logger('my_class',config) 
    '''

    def setup_class_logger(
            self, 
            name: str, 
            config: dict,
            parent_key: Optional[str]
    ):
        ''' 
        setup logger with configuration from config dict

        Args:
            name: Logger name (e.g. 'data_spliter')
            config: configuration dictionary
            parent_key: optional key provided to extract logging config

        Returns:
            configured logging instance
        '''
        try:
            # extract logging config from parent if provided
            if parent_key and parent_key in config:
                log_config = config[parent_key].get('logging',{})
            else:
                log_config = config.get('logging',{})


            # setup the log directory
            log_dir = Path(log_config.get('log_dir','logs/'))
            ensure_directory(log_dir)

            # create logger
            logger = setup_logger(
                name=name,
                log_dir=log_dir,
                log_level=log_config.get('log_level','INFO'),
                max_bytes=log_config.get('max_bytes', 10485760),
                backup_count=log_config.get('backup_count',7)
            )
            
            return logger
        
        except Exception as e:
            # fallback to basic logging if setup fails
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(name)
            logger.error(f'Error setting up logger: {e}')
            return logger