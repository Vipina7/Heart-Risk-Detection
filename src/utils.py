import os
import sys
from src.exception import CustomException
from src.logger import logging
import pickle
import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Saved preprocessing object")

    except Exception as e:
        raise CustomException(e, sys)