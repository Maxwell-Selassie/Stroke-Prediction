
from pathlib import Path
from utils import ensure_directory, setup_logger

class LoggerMixin:
    """Mixin to add logging capabilities to any class"""
    
    def setup_class_logger(self, name, config):
        log_config = config.get('logging', {})
        log_dir = Path(log_config.get('log_dir', 'logs/'))
        ensure_directory(log_dir)
        
        return setup_logger(
            name=name,
            log_dir=log_dir,
            log_level=log_config.get('log_level', 'INFO'),
            max_bytes=log_config.get('max_bytes', 10485760),
            backup_count=log_config.get('backup_count', 7)
        )

#