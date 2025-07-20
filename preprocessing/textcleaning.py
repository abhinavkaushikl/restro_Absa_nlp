import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
from pathlib import Path
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RestaurantTextCleaner:
    """Comprehensive text cleaner for restaurant ABSA data"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # ABSA-specific: Preserve negations and sentiment indicators
        self.preserve_words = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'noone', 'none', 
            'neither', 'nobody', 'cannot', "can't", "won't", "shouldn't", 
            "wouldn't", "couldn't", "doesn't", "don't", "isn't", "aren't",
            'but', 'however', 'although', 'though', 'yet', 'still'
        }
        
        # Restaurant-specific terms to remove
        self.restaurant_generic_terms = {
            'restaurant', 'place', 'spot', 'location', 'establishment',
            'venue', 'joint', 'eatery', 'dining', 'dine'
        }
        
        # Update stopwords - remove preserved words, add generic terms
        self.stop_words = (self.stop_words - self.preserve_words) | self.restaurant_generic_terms
        
        # Contraction dictionary
        self.contractions_dict = {
            "ain't": "are not", "aren't": "are not", "can't": "cannot", 
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not", 
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
            "i've": "i have", "isn't": "is not", "it'd": "it would", 
            "it'll": "it will", "it's": "it is", "let's": "let us",
            "shouldn't": "should not", "that's": "that is", "there's": "there is",
            "they'd": "they would", "they'll": "they will", "they're": "they are",
            "they've": "they have", "we'd": "we would", "we're": "we are",
            "we've": "we have", "weren't": "were not", "what's": "what is",
            "where's": "where is", "who's": "who is", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
            "you're": "you are", "you've": "you have", "wasn't": "was not"
        }

    def load_data(self, file_path):
        """Load restaurant ABSA data from CSV"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def remove_html_xml_tags(self, text):
        """1. Essential: Remove HTML/XML tags"""
        if pd.isna(text):
            return ""
        text = str(text)
        # Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        return text

    def remove_urls_emails(self, text):
        """1. Essential: Remove URLs and emails"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove social media mentions and hashtags
        text = re.sub(r'@\w+|#\w+', ' ', text)
        
        return text

    def handle_special_characters(self, text):
        """1. Essential: Handle special characters - preserve sentiment indicators"""
        # Keep important punctuation for sentiment: .,!?;:
        important_punct = '.,!?;:'
        
        # Remove other punctuation except important ones
        chars_to_remove = string.punctuation.translate(str.maketrans('', '', important_punct))
        text = re.sub(f'[{re.escape(chars_to_remove)}]', ' ', text)
        
        # Clean up excessive punctuation but preserve sentiment
        text = re.sub(r'[!]{3,}', '!!', text)  # Keep double exclamation for strong sentiment
        text = re.sub(r'[?]{3,}', '??', text)  # Keep double question for emphasis
        text = re.sub(r'[.]{3,}', '...', text)  # Keep ellipsis
        
        # Remove numbers (they're usually not important for sentiment)
        text = re.sub(r'\d+', ' ', text)
        
        return text

    def expand_contractions(self, text):
        """1. Essential: Expand contractions"""
        text_lower = text.lower()
        for contraction, expansion in self.contractions_dict.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(contraction) + r'\b'
            text_lower = re.sub(pattern, expansion, text_lower)
        return text_lower

    def normalize_case(self, text):
        """1. Essential: Case normalization"""
        return text.lower()

    def normalize_whitespace(self, text):
        """1. Essential: Whitespace normalization"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def handle_repeated_characters(self, text):
        """2. ABSA-Specific: Handle repeated characters while preserving emphasis"""
        # Reduce repeated characters but keep some emphasis
        # "sooooo good" -> "soo good" (keeps some emphasis)
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        return text

    def preserve_aspect_context(self, text, aspects_json):
        """2. ABSA-Specific: Ensure aspect terms are preserved"""
        if pd.isna(aspects_json) or aspects_json == '[]':
            return text
        
        try:
            aspects = json.loads(aspects_json)
            aspect_terms = [aspect['term'].lower() for aspect in aspects if 'term' in aspect]
            
            # Create a set of important aspect-related words to preserve
            aspect_words = set()
            for term in aspect_terms:
                aspect_words.update(term.split())
            
            # These words should not be removed even if they're in stopwords
            self.preserve_aspect_words = aspect_words
            
        except (json.JSONDecodeError, KeyError):
            pass
        
        return text

    def remove_stopwords_carefully(self, tokens):
        """3. Optional: Careful stopword removal for restaurant reviews"""
        if not tokens:
            return []
        
        # Keep negations, sentiment words, and aspect-related terms
        filtered_tokens = []
        for token in tokens:
            if (token not in self.stop_words or 
                len(token) <= 2 or  # Keep short words (often important)
                token in self.preserve_words or
                hasattr(self, 'preserve_aspect_words') and token in self.preserve_aspect_words):
                filtered_tokens.append(token)
        
        return filtered_tokens

    def tokenize_text(self, text):
        """3. Optional: Tokenization"""
        if not text or pd.isna(text):
            return []
        
        tokens = word_tokenize(str(text))
        # Filter out empty tokens and single punctuation
        tokens = [token for token in tokens if token.strip() and len(token) > 1 or token in '!?.,']
        return tokens

    def lemmatize_tokens(self, tokens):
        """3. Optional: Lemmatization"""
        if not tokens:
            return []
        
        # Skip lemmatization for preserved words to maintain sentiment
        lemmatized = []
        for token in tokens:
            if token in self.preserve_words or token in '!?.,':
                lemmatized.append(token)
            else:
                lemmatized.append(self.lemmatizer.lemmatize(token))
        
        return lemmatized

    def clean_sentence(self, sentence, aspects_json=None):
        """Complete cleaning pipeline for a single sentence"""
        if pd.isna(sentence):
            return {
                'sentence_cleaned': "",
                'sentence_tokens': [],
                'sentence_no_stopwords': [],
                'sentence_lemmatized': [],
                'sentence_final': ""
            }
        
        # Step 1: Essential cleaning operations
        text = self.remove_html_xml_tags(sentence)
        text = self.remove_urls_emails(text)
        text = self.handle_special_characters(text)
        text = self.expand_contractions(text)
        text = self.normalize_case(text)
        text = self.handle_repeated_characters(text)
        text = self.normalize_whitespace(text)
        
        # Step 2: ABSA-specific considerations
        text = self.preserve_aspect_context(text, aspects_json)
        
        # Step 3: Optional steps
        tokens = self.tokenize_text(text)
        tokens_no_stopwords = self.remove_stopwords_carefully(tokens)
        tokens_lemmatized = self.lemmatize_tokens(tokens_no_stopwords)
        final_text = ' '.join(tokens_lemmatized)
        
        return {
            'sentence_cleaned': text,
            'sentence_tokens': tokens,
            'sentence_no_stopwords': tokens_no_stopwords,
            'sentence_lemmatized': tokens_lemmatized,
            'sentence_final': final_text
        }

    def process_dataframe(self, df):
        """Process entire dataframe"""
        logger.info("Starting text cleaning pipeline...")
        
        # Apply cleaning to each sentence
        cleaning_results = []
        for idx, row in df.iterrows():
            sentence = row.get('sentence', '')
            aspects = row.get('aspects', '[]')
            
            result = self.clean_sentence(sentence, aspects)
            cleaning_results.append(result)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} sentences")
        
        # Add cleaned columns to dataframe
        for key in cleaning_results[0].keys():
            df[key] = [result[key] for result in cleaning_results]
        
        # Calculate cleaning statistics
        self.calculate_stats(df)
        
        logger.info("Text cleaning pipeline completed!")
        return df

    def calculate_stats(self, df):
        """Calculate and display cleaning statistics"""
        original_lengths = df['sentence'].str.len()
        cleaned_lengths = df['sentence_final'].str.len()
        
        logger.info("=== Text Cleaning Statistics ===")
        logger.info(f"Total sentences processed: {len(df)}")
        logger.info(f"Average original length: {original_lengths.mean():.1f} characters")
        logger.info(f"Average cleaned length: {cleaned_lengths.mean():.1f} characters")
        logger.info(f"Length reduction: {((original_lengths.mean() - cleaned_lengths.mean()) / original_lengths.mean() * 100):.1f}%")
        
        # Word count statistics
        original_words = df['sentence'].str.split().str.len()
        cleaned_words = df['sentence_final'].str.split().str.len()
        logger.info(f"Average original words: {original_words.mean():.1f}")
        logger.info(f"Average cleaned words: {cleaned_words.mean():.1f}")
        
        # Show sample transformations
        logger.info("\n=== Sample Transformations ===")
        for i in range(min(3, len(df))):
            logger.info(f"\nSample {i+1}:")
            logger.info(f"Original: {df.iloc[i]['sentence']}")
            logger.info(f"Cleaned:  {df.iloc[i]['sentence_final']}")

    def save_cleaned_data(self, df, output_path):
        """Save cleaned data to CSV"""
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Cleaned data saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

def process_restaurant_data(csv_file_path):
    """Main function to process restaurant ABSA data"""
    cleaner = RestaurantTextCleaner()
    
    # Load data
    df = cleaner.load_data(csv_file_path)
    if df is None:
        return None
    
    # Process data
    df_cleaned = cleaner.process_dataframe(df)
    print("Data cleaning completed. Cleaned data shape:", df_cleaned.shape)
    
    # Save cleaned data
    output_path = csv_file_path.replace('.csv', '_cleaned.csv')
    cleaner.save_cleaned_data(df_cleaned, output_path)
    
    return df_cleaned

# Decorator function to automatically run with restaurant_absa_data.csv
def restaurant_absa_data_csv(func):
    """Decorator to automatically run with restaurant_absa_data.csv"""
    def wrapper():
        # Find the CSV file
        project_root = Path(__file__).parent.parent
        csv_path = project_root / 'data' / 'processed' / 'restaurant_absa_data.csv'
        process_restaurant_data(str(csv_path))

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Define the input CSV file path
    csv_path = project_root / 'data' / 'processed' / 'restaurant_absa_data.csv'
    
    # Process the restaurant data
    process_restaurant_data(str(csv_path))

if __name__ == "__main__":
    main()

