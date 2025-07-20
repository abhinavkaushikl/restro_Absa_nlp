from fastapi import FastAPI
from pydantic import BaseModel
from preprocessing.absa_pipeline import AspectExtractor
from models.sentimentanalyzer import SentimentTrainer
import joblib
from pathlib import Path

app = FastAPI(
    title="Restaurant Review Complete Analysis",
    description="Aspect Extraction and Trained Sentiment Analysis",
    version="1.0.0"
)

# Load models
aspect_extractor = AspectExtractor()
model_dir = Path('models/trained')
vectorizer = joblib.load(model_dir / 'vectorizer.pkl')
sentiment_model = joblib.load(model_dir / 'sentiment_model.pkl')

class ReviewText(BaseModel):
    text: str

@app.post("/analyze_review")
def analyze_review(review: ReviewText):
    # Step 1: Extract aspects using ABSA pipeline
    aspects = aspect_extractor.extract_aspects(review.text)
    
    # Step 2: Analyze sentiment for each aspect using trained model
    analysis_results = []
    for aspect in aspects:
        # Get context
        context = aspect_extractor.get_aspect_window(review.text, aspect)
        
        # Get sentiment prediction
        text_vec = vectorizer.transform([context])
        sentiment = sentiment_model.predict(text_vec)[0]
        probabilities = sentiment_model.predict_proba(text_vec)[0]
        
        analysis_results.append({
            "aspect": aspect,
            "sentiment": "positive" if sentiment == 1 else "negative" if sentiment == 0 else "neutral",
            "probability": float(max(probabilities)),
            "context": context
        })
    
    return {
        "text": review.text,
        "aspects_found": len(aspects),
        "analysis": analysis_results
    }

@app.get("/")
def read_root():
    return {
        "message": "Restaurant Review Analysis API",
        "endpoints": {
            "/analyze_review": "POST - Full review analysis with aspects and sentiments"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
