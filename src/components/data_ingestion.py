"""
This file handles reading raw data, splitting it into training 
and testing sets, saving these datasets, and logging the process, while 
handling errors with custom exceptions.
"""

import os
import sys
from src.exception import CustomException  # Importing CustomException from exception.py within src
from src.logger import logging  # Importing logging setup from logger.py within src
import pandas as pd

from sklearn.model_selection import train_test_split  # Importing train_test_split from sklearn
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation  # Importing DataTransformation class from data_transformation.py within src.components
from src.components.model_trainer import ModelTrainer  # Importing ModelTrainer class from model_trainer.py within src.components

@dataclass # automatically generates methods like __init__, __repr__, __eq__, and __hash__
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    
    Attributes:
    train_data_path (str): Path to save the training data CSV file.
    test_data_path (str): Path to save the testing data CSV file.
    raw_data_path (str): Path to save the raw data CSV file.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

@dataclass
class DataIngestion:
    """
    Class for managing the data ingestion process.
    """
    """
    Initializes DataIngestion with default configuration.
    """
    # ingestion_config is an object of class DataIngestionConfig
    ingestion_config: DataIngestionConfig = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process.

        Returns:
        tuple: Paths to the training and testing data CSV files.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')  # Read data from CSV file into a DataFrame
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)  # Create directories for saving data

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)  # Save raw data to CSV file

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)  # Split data into training and testing sets

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)  # Save training set to CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)  # Save testing set to CSV file

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)  # Raise CustomException if an error occurs during data ingestion

if __name__ == "__main__":
    obj = DataIngestion()  # Create DataIngestion object
    train_data, test_data = obj.initiate_data_ingestion()  # Initiate data ingestion process and get paths to training and testing data

    data_transformation = DataTransformation()  # Create DataTransformation object
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)  # Initiate data transformation process

    model_trainer = ModelTrainer()  # Create ModelTrainer object
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))  # Initiate model training process and print results
