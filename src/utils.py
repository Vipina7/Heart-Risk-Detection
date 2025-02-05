import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import dill
from tqdm import tqdm

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Saved preprocessing object")

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    try:
        report_train = {}
        report_test = {}

        strat_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        logging.info('Model training initiated')
        for i in tqdm(range(len(list(models)))):
            model = list(models.values())[i]
            params=param[list(models.keys())[i]]

            gs = GridSearchCV(model,params,cv=strat_kf, scoring = ['accuracy', 'f1'], refit = 'accuracy', n_jobs=-1)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = [
            precision_score(y_train, y_train_pred),
            recall_score(y_train, y_train_pred),
            f1_score(y_train, y_train_pred),
            accuracy_score(y_train, y_train_pred)
        ]

            test_model_score = [
            precision_score(y_test, y_test_pred),
            recall_score(y_test, y_test_pred),
            f1_score(y_test, y_test_pred),
            accuracy_score(y_test, y_test_pred)
        ]
            report_train[list(models.keys())[i]] = train_model_score
            report_test[list(models.keys())[i]] = test_model_score
        
        logging.info('Model performance reports generated')
        df_report_test = pd.DataFrame.from_dict(report_test, orient='index', columns=['precision', 'recall', 'f1', 'accuracy'])
        df_report_train = pd.DataFrame.from_dict(report_train, orient='index', columns= ['precision', 'recall', 'f1', 'accuracy'])

        df_report_test.to_csv(os.path.join('artifacts', 'model_performance.csv'), index=True)

        return df_report_test
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)