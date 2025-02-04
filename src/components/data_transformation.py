import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataTransformationConfig:
    transformation_obj_path:str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor_object(self):
        try:
            logging.info("Data preprocessing initiated")
            numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
            categorical_cols = ['gender', 'origin', 'cp', 'fbs', 'restecg', 'exang']

            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder())
            ])

            numerical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('encoder', OneHotEncoder())
            ])

            preprocessor = ColumnTransformer([
                ('num_cols', numerical_transformer, numerical_cols),
                ('cat_cols', categorical_transformer, categorical_cols)
            ])
            logging.info('Preprocessor object obtained')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Importing train and test sets")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = train_df.drop(columns = ['slope','ca','thal','id'], axis=1).reset_index(drop=True)
            test_df = test_df.drop(columns = ['slope','ca','thal','id'], axis=1).reset_index(drop=True)
        
        except Exception as e:
            raise CustomException(e,sys)