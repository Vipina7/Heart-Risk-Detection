import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def intiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('Notebook\Data\heart_disease_uci.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            logging.info('Artifacts folder created')

            df = df.rename({'num':'heart_disease', 'dataset':'origin', 'sex':'gender'}, axis=1)
            df = df.drop(columns=['slope','ca','thal','id']).reset_index(drop=True)
            df['heart_disease'] = df['heart_disease'].apply(lambda x: 1 if x>0 else 0)
            logging.info('Renaming and dropping the necessary columns')

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index = False, header = True)
            logging.info('Ingestion of data is completed')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    data_obj = DataIngestion()
    train_path, test_path = data_obj.intiate_data_ingestion()

    transform_obj = DataTransformation()
    X_train, X_test, y_train, y_test = transform_obj.initiate_data_transformation(train_path=train_path, test_path=test_path)

    trainer_obj = ModelTrainer()
    print(trainer_obj.initiate_model_training(X_train, X_test, y_train, y_test))