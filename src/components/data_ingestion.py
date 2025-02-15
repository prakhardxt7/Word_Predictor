import os
import sys
import nltk
import pandas as pd
from nltk.corpus import gutenberg
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

nltk.download('gutenberg')

data = gutenberg.raw('shakespeare-hamlet.txt')

# Save the raw text to a file
with open('hamlet.txt', 'w') as file:
    file.write(data)

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.txt')
    test_data_path: str = os.path.join('artifacts', 'test.txt')
    raw_data_path: str = os.path.join('artifacts', 'raw.txt')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entering the data ingestion method or component!')
        try:
            # Ensure artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            with open(self.ingestion_config.raw_data_path, 'w') as file:
                file.write(data)
            
            logging.info('Train Test Split Initiated!')
            text_lines = data.split('\n')
            train_set, test_set = train_test_split(text_lines, test_size=0.2, random_state=42)
            
            # Save train and test sets
            with open(self.ingestion_config.train_data_path, 'w') as file:
                file.write('\n'.join(train_set))
            with open(self.ingestion_config.test_data_path, 'w') as file:
                file.write('\n'.join(test_set))
            
            logging.info('Ingestion of data is completed!')
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")
