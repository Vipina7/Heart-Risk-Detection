import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Loading the model and preprocessor object")
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("Preprocessing the input features")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making predictions successful")
            prediction = model.predict(data_scaled)
            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 age,
                 gender,
                 origin,
                 cp,
                 trestbps,
                 chol,
                 fbs,
                 restecg,
                 thalch,
                 exang,
                 oldpeak):
        
        self.age = age
        self.gender = gender
        self.origin = origin
        self.cp = cp
        self.trestbs = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalch = thalch
        self.exang = exang
        self.oldpeak = oldpeak

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "gender": [self.gender],
                "origin": [self.origin],
                "cp": [self.cp],
                "trestbps": [self.trestbs],  
                "chol": [self.chol],
                "fbs": [self.fbs],
                "restecg": [self.restecg],
                "thalch": [self.thalch],
                "exang": [self.exang],
                "oldpeak": [self.oldpeak]
            }
            logging.info("Dataframe ready for prediction")

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
