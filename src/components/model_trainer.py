import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object
from dataclasses import dataclass

from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainerConfig:
    trained_model_path:str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(use_label_encoder=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Naive Bayes": GaussianNB(),
                "KNN":KNeighborsClassifier(),
                "SVC":SVC()
                }
            
            param = {
                "Decision Tree": {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random']
                },
    
                "Random Forest": {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
                },
    
                "Gradient Boosting": {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 1.0],
                'criterion': ['friedman_mse', 'squared_error']
                },
    
                "Logistic Regression": {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'C': [0.1, 1.0, 10],
                'solver': ['liblinear', 'lbfgs', 'saga'],
                'max_iter': [100, 200, 300]
                },
    
                "XGBClassifier": {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'objective': ['binary:logistic'],
                'eval_metric': ['logloss']
                },
    
                "AdaBoost Classifier": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.5, 1.0, 1.5],
                'algorithm': ['SAMME', 'SAMME.R']
                },

                "Naive Bayes": {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                },
    
                "KNN": {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
                },

                "SVC": {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'degree': [2, 3, 4]
                }
                }
            
            report = evaluate_models(X_train, X_test, y_train, y_test, models, param)

            best_model_score = report['accuracy'].max()
            best_model_name = report[report['accuracy']== best_model_score].index[0]
            best_model =models[best_model_name]
            
            save_object(
                file_path = self.model_trainer_config.trained_model_path,
                obj = best_model
            )
            logging.info('Saved the model')

            predicted=best_model.predict(X_test)

            return classification_report(y_test, predicted)
        
        except Exception as e:
            raise CustomException(e, sys)