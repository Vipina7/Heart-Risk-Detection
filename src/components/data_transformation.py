import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
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
                ('encoder', StandardScaler())
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

            logging.info('Obtaining preprocessor object')
            preprocessor_obj = self.get_preprocessor_object()

            X_train = train_df.drop(columns = ['heart_disease'], axis=1)
            y_train = train_df['heart_disease']

            X_test = test_df.drop(columns=['heart_disease'], axis=1)
            y_test = test_df['heart_disease']
            logging.info("Obtained independent and dependent features")

            X_train_scaled = preprocessor_obj.fit_transform(X_train)
            X_test_scaled = preprocessor_obj.transform(X_test)
            logging.info("Data transformation complete")

            save_object(
                file_path=self.data_transformation_config.transformation_obj_path,
                obj=preprocessor_obj)

            return (
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test
            )
        
        except Exception as e:
            raise CustomException(e,sys)