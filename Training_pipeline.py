from pathlib import Path
import logging
import pandas as pd
from preprocessing.Dataloader import RestaurantDataLoader
from preprocessing.textcleaning import RestaurantTextCleaner
from preprocessing.absa_pipeline import ABSAPipeline
from models.sentimentanalyzer import SentimentTrainer
from utils.config_reader import ConfigReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_sequential_pipeline():
    logger.info("Starting Sequential Pipeline...")
    
    # Initialize components
    config = ConfigReader()
    data_loader = RestaurantDataLoader(config_reader=config)
    text_cleaner = RestaurantTextCleaner()
    absa_pipeline = ABSAPipeline()
    sentiment_trainer = SentimentTrainer()

    # Step 1: Data Loading
    logger.info("Step 1: Loading Data...")
    dataset_path = config.get_dataset_path()
    data_records = data_loader.load_restaurant_xml(dataset_path)
    df = data_loader.create_dataframe(data_records)
    
    # Ensure both sentence and text columns exist
    df['sentence'] = df['text'] if 'text' in df.columns else df['sentence']
    df['text'] = df['sentence']
    
    processed_dir = Path(dataset_path).parent / 'processed'
    processed_dir.mkdir(exist_ok=True)
    processed_path = processed_dir / 'restaurant_absa_data.csv'
    data_loader.save_processed_data(df, processed_path)
    logger.info("Data loading completed")

    # Step 2: Text Cleaning
    logger.info("Step 2: Cleaning Text...")
    cleaned_df = text_cleaner.process_dataframe(df)
    # Maintain both column names for compatibility
    cleaned_df['sentence'] = cleaned_df['sentence_final']
    cleaned_df['text'] = cleaned_df['sentence_final']
    cleaned_path = processed_dir / 'restaurant_absa_data_cleaned.csv'
    text_cleaner.save_cleaned_data(cleaned_df, cleaned_path)
    logger.info("Text cleaning completed")

    # Step 3: ABSA Processing
    logger.info("Step 3: Running ABSA Pipeline...")
    output_dir = processed_dir / 'absa_output'
    output_dir.mkdir(exist_ok=True)
    absa_results = absa_pipeline.process_data(
        input_file=str(cleaned_path),
        output_dir=str(output_dir)
    )
    
    # Ensure ABSA results have the required column
    if isinstance(absa_results, pd.DataFrame):
        absa_results['sentence'] = absa_results['text']
        absa_results_path = output_dir / 'absa_results.csv'
        absa_results.to_csv(absa_results_path, index=False)
    else:
        absa_results_path = output_dir / 'absa_results.csv'
    
    logger.info("ABSA processing completed")

    # Step 4: Sentiment Analysis Training
    logger.info("Step 4: Training Sentiment Analyzer...")
    sentiment_score = sentiment_trainer.train(str(absa_results_path))
    logger.info(f"Sentiment training completed with score: {sentiment_score}")

    return {
        'processed_data': processed_path,
        'cleaned_data': cleaned_path,
        'absa_results': absa_results,
        'sentiment_score': sentiment_score
    }


def main():
    results = run_sequential_pipeline()
    logger.info("Sequential pipeline completed successfully!")
    logger.info(f"Final Results: {results}")

if __name__ == "__main__":
    main()
