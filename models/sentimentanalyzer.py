import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

class SentimentTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(random_state=42)
        
    def prepare_data(self, df):
        # Extract text and aspects with sentiments
        texts = []
        sentiments = []
        contexts = []
        
        for _, row in df.iterrows():
            text = row['sentence']
            aspects = eval(row['aspects'])
            for aspect in aspects:
                context = self.get_context(text, aspect)
                texts.append(context)
                sentiments.append(self.get_sentiment_label(context))
                contexts.append(context)
                
        return texts, sentiments, contexts
    
    def get_context(self, text, aspect, window_size=10):
        words = text.split()
        try:
            aspect_idx = [i for i, word in enumerate(words) if aspect in word][0]
            start = max(0, aspect_idx - window_size)
            end = min(len(words), aspect_idx + window_size)
            return ' '.join(words[start:end])
        except IndexError:
            return text
    
    def get_sentiment_label(self, text):
        # Simple rule-based labeling for initial training
        positive_words = {'good', 'great', 'excellent', 'delicious', 'amazing'}
        negative_words = {'bad', 'poor', 'terrible', 'slow', 'disappointing'}
        
        text_words = set(text.lower().split())
        pos_count = len(text_words.intersection(positive_words))
        neg_count = len(text_words.intersection(negative_words))
        
        if pos_count > neg_count:
            return 1
        elif neg_count > pos_count:
            return 0
        return -1  # neutral
    
    def train(self, data_path):
        print("Loading data...")
        df = pd.read_csv(data_path)
        
        print("Preparing training data...")
        texts, sentiments, contexts = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, sentiments, test_size=0.2, random_state=42
        )
        
        print("Training model...")
        # Transform text data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        score = self.model.score(X_test_vec, y_test)
        print(f"Model accuracy: {score:.2f}")
        
        # Save model
        self.save_model()
        
        return score
    
    def predict_sentiment(self, text):
        text_vec = self.vectorizer.transform([text])
        sentiment = self.model.predict(text_vec)[0]
        proba = self.model.predict_proba(text_vec)[0]
        
        return {
            'sentiment': 'positive' if sentiment == 1 else 'negative' if sentiment == 0 else 'neutral',
            'probability': max(proba),
            'context': text
        }
    
    def save_model(self):
        model_dir = Path('models/trained')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.vectorizer, model_dir / 'vectorizer.pkl')
        joblib.dump(self.model, model_dir / 'sentiment_model.pkl')
        print("Model saved successfully!")

def main():
    trainer = SentimentTrainer()
       # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Define the input CSV file path
    csv_path = project_root / 'data' / 'processed' / 'absa_results.csv'
    
   
    trainer.train(csv_path)

if __name__ == "__main__":
    main()
