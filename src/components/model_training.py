import os
from pyexpat import model
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_objects
from src.utils import model_evaluates
from sklearn.metrics import classification_report,accuracy_score



from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)



from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from catboost import CatBoostClassifier
from dataclasses import dataclass

@dataclass
class modelTrainerConfig():
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class modelTrainer():
    def __init__(self):
        self.model_path=modelTrainerConfig()

    try:

        def initiate_model_trainer(self,train_arr,test_arr):

            logging.info("Splitting training and test input data")  

            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            logging.info("Splitting done successfully")

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGB Classifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier()
            }

            logging.info("Models defined successfully")

            model_evu : dict = model_evaluates(x_train,y_train,x_test,y_test,models)

            logging.info(f"evaluation of all model : {model_evu}")

            best_model_score = max(sorted(model_evu.values()))   
            best_model_name = list(model_evu.keys())[list(model_evu.values()).index(best_model_score)]   
            best_model = models[best_model_name]

            logging.info(f"Best model found, model name is {best_model_name} with accuracy score {best_model_score}")

            save_objects(
                file_path=self.model_path.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved successfully")

            return (
                self.model_path.trained_model_file_path
            )

        
    except Exception as e :
        raise CustomException(e,sys)


            


            


            



