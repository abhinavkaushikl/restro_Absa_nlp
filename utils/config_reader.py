import yaml
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigReader:
    """Config reader for loading YAML configuration"""
    
    def __init__(self, config_path='config/config.yaml'):
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            logger.info(f"Config loaded from: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return None
    
    def get_dataset_path(self):
        """Get dataset file path from config"""
        if self.config is None:
            return None
        
        try:
            dataset_path = self.config['dataset']['file_path']
            # Clean raw string format
            dataset_path = dataset_path.strip('r"')
            return dataset_path
        except KeyError:
            logger.error("Dataset file path not found in config")
            return None
