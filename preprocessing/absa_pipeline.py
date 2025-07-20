import spacy
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, List, Set

class AspectExtractor:
    def __init__(self, domain_config: Dict = None):
        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load domain-specific configuration or use default
        self.domain_config = domain_config or {
            'aspect_terms': {
                'general': {'quality', 'price', 'service', 'value', 'experience'},
                'specific': set(),  # Domain-specific terms
            },
            'multi_word_patterns': [
                ['NOUN', 'NOUN'],
                ['ADJ', 'NOUN'],
                ['NOUN', 'ADP', 'NOUN']
            ],
            'relevant_ner_labels': {'PRODUCT', 'ORG'}
        }

    def get_aspect_window(self, text: str, aspect: str, window_size: int = 6) -> str:
        """Extract context window around aspect term"""
        doc = self.nlp(text)
        aspect_tokens = aspect.lower().split()
        text_tokens = [token.text.lower() for token in doc]
        
        for i in range(len(text_tokens) - len(aspect_tokens) + 1):
            if text_tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                start = max(0, i - window_size)
                end = min(len(text_tokens), i + len(aspect_tokens) + window_size)
                return ' '.join(text_tokens[start:end])
        return text

    def extract_noun_phrases(self, doc) -> List[str]:
        """Extract base noun phrases"""
        return [chunk.text.lower() for chunk in doc.noun_chunks]

    def extract_multi_word_aspects(self, doc) -> List[str]:
        """Extract aspects based on dependency patterns"""
        aspects = []
        for token in doc:
            if token.dep_ in {'compound', 'amod'} and token.head.pos_ == 'NOUN':
                aspect = ' '.join([token.text, token.head.text]).lower()
                aspects.append(aspect)
        return aspects

    def extract_ner_aspects(self, doc) -> List[str]:
        """Extract relevant named entities"""
        return [ent.text.lower() for ent in doc.ents 
                if ent.label_ in self.domain_config['relevant_ner_labels']]

    def match_domain_terms(self, text: str) -> List[str]:
        """Match against domain-specific terms"""
        matched = []
        text_lower = text.lower()
        all_terms = (self.domain_config['aspect_terms']['general'] | 
                    self.domain_config['aspect_terms']['specific'])
        
        for term in all_terms:
            if term in text_lower:
                matched.append(term)
        return matched

    def extract_aspects(self, text: str) -> List[str]:
        """Main aspect extraction pipeline"""
        if not text or pd.isna(text):
            return []
            
        doc = self.nlp(str(text))
        aspects = set()
        
        # Combine all extraction methods
        aspects.update(self.extract_noun_phrases(doc))
        aspects.update(self.extract_multi_word_aspects(doc))
        aspects.update(self.extract_ner_aspects(doc))
        aspects.update(self.match_domain_terms(text))
        
        return list(self.filter_aspects(aspects))

    def filter_aspects(self, aspects: Set[str]) -> Set[str]:
        """Filter aspects based on rules"""
        return {aspect for aspect in aspects 
                if 2 <= len(aspect.split()) <= 4 
                and len(aspect) >= 3}

class ABSAPipeline:
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.aspect_extractor = AspectExtractor(self.config.get('domain_config'))
        
    def load_config(self, config_path: str) -> Dict:
        """Load domain configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def process_data(self, input_file: str, output_dir: str):
        """Process data through ABSA pipeline"""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and process data
        df = pd.read_csv(input_file)
        results = []
        
        print(f"Processing {len(df)} records...")
        
        for idx, row in df.iterrows():
            text = row['text']
            
            # Extract aspects
            aspects = self.aspect_extractor.extract_aspects(text)
            
            result = {
                'id': idx,
                'text': text,
                'aspects': aspects
            }
            results.append(result)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} records")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        self.save_results(results_df, output_path)
        
        return results_df

    def save_results(self, df: pd.DataFrame, output_path: Path):
        """Save results and statistics"""
        # Save main results
        df.to_csv(output_path / 'absa_results.csv', index=False)
        
        # Generate and save statistics
        stats = self.generate_statistics(df)
        with open(output_path / 'absa_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)

    def generate_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate analysis statistics"""
        stats = {
            'total_records': len(df),
            'records_with_aspects': len(df[df['aspects'].str.len() > 0]),
            'unique_aspects': len(set([
                aspect for aspects in df['aspects'] 
                for aspect in aspects if aspects
            ])),
            'aspect_frequency': defaultdict(int)
        }
        
        # Calculate aspect frequencies
        for aspects in df['aspects']:
            if aspects:
                for aspect in aspects:
                    stats['aspect_frequency'][aspect] += 1
        
        return stats

def main():
    # Initialize pipeline
    pipeline = ABSAPipeline()
    
    # Define input and output paths
    input_file = 'data/processed/absa_results.csv'
    output_dir = 'data/processed/absa_results'
    
    # Process data
    results = pipeline.process_data(input_file, output_dir)
    print("ABSA pipeline completed successfully!")

if __name__ == "__main__":
    main()
