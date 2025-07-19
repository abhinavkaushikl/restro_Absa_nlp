import xml.etree.ElementTree as ET
import pandas as pd
import json
import logging
from pathlib import Path
from utils.config_reader import ConfigReader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RestaurantDataLoader:
    """Data loader for Restaurant ABSA dataset with aspect and category extraction"""
    
    def __init__(self, config_reader):
        self.config_reader = config_reader
    
    def load_restaurant_xml(self, file_path):
        """Load and parse restaurant XML file"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            logger.info(f"Successfully parsed XML file: {file_path}")
            
            # Extract all sentences
            sentences = root.findall('.//sentence')
            logger.info(f"Found {len(sentences)} sentences in the dataset")
            
            data_records = []
            
            for sentence in sentences:
                record = self.extract_sentence_data(sentence)
                if record:
                    data_records.append(record)
            
            logger.info(f"Extracted {len(data_records)} valid records")
            return data_records
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading XML: {e}")
            return None
    
    def extract_sentence_data(self, sentence):
        """Extract data from a single sentence element"""
        try:
            # Basic sentence information
            sentence_id = sentence.get('id', '')
            text = sentence.find('text')
            text_content = text.text.strip() if text is not None and text.text else ''
            
            if not text_content:
                return None
            
            # Initialize record
            record = {
                'id': sentence_id,
                'sentence': text_content,
                'aspects': self.extract_aspect_terms(sentence),
                'aspectCategories': self.extract_aspect_categories(sentence)
            }
            
            return record
            
        except Exception as e:
            logger.error(f"Error extracting sentence data: {e}")
            return None
    
    def extract_aspect_terms(self, sentence):
        """Extract aspect terms and their details"""
        aspect_terms_elem = sentence.find('aspectTerms')
        
        aspects = []
        
        if aspect_terms_elem is not None:
            for aspect_term in aspect_terms_elem.findall('aspectTerm'):
                term = aspect_term.get('term', '').strip()
                polarity = aspect_term.get('polarity', 'neutral')
                from_pos = aspect_term.get('from', '')
                to_pos = aspect_term.get('to', '')
                
                if term:
                    aspect_data = {
                        'term': term,
                        'polarity': polarity,
                        'start': from_pos,
                        'end': to_pos
                    }
                    aspects.append(aspect_data)
        
        return json.dumps(aspects)
    
    def extract_aspect_categories(self, sentence):
        """Extract aspect categories and their details"""
        aspect_categories_elem = sentence.find('aspectCategories')
        
        categories = []
        
        if aspect_categories_elem is not None:
            for aspect_category in aspect_categories_elem.findall('aspectCategory'):
                category = aspect_category.get('category', '').strip()
                polarity = aspect_category.get('polarity', 'neutral')
                
                if category:
                    category_data = {
                        'category': category,
                        'polarity': polarity
                    }
                    categories.append(category_data)
        
        return json.dumps(categories)
    
    def create_dataframe(self, data_records):
        """Convert extracted records to pandas DataFrame"""
        if not data_records:
            logger.warning("No data records to convert")
            return None
        
        df = pd.DataFrame(data_records)
        logger.info(f"Created DataFrame with shape: {df.shape}")
        return df
    
    def save_processed_data(self, df, output_path=None):
        """Save processed data to CSV"""
        if df is None or df.empty:
            logger.warning("No data to save")
            return
        
        if output_path is None:
            output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'restaurant_absa_data.csv'
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Dataset saved to: {output_path}")

# Usage example
def main():
    """Test the RestaurantDataLoader"""
    config_reader = ConfigReader()
    loader = RestaurantDataLoader(config_reader)
    
    # Get dataset path
    dataset_path = config_reader.get_dataset_path()
    if dataset_path is None:
        logger.error("Dataset path is not available")
        return
    
    # Load and process dataset
    data_records = loader.load_restaurant_xml(dataset_path)
    if data_records is None:
        logger.error("Failed to load dataset")
        return
    
    # Create DataFrame
    df = loader.create_dataframe(data_records)
    if df is not None:
        # Save processed data
        loader.save_processed_data(df)

if __name__ == "__main__":
    main()
