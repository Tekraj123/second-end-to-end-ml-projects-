import os
import pandas as pd
import numpy as np
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import classification_report,accuracy_score

from src.logger import logging

# os.path.dirname(file_path) → extracts only the directory part of that path, removing the actual file name.

def save_objects(obj,file_path):  
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)


def model_evaluates(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        logging.info(f"akhdfkadhdakhkdhaskjhgsjkahjkgggggggggggggggggggggggggggggggggggggggggggg{y_train}")
        for i in range(len(models)):
            model=list(models.values())[i]  
            model.fit(x_train,y_train)       
            prd_value = model.predict (x_test)   
            accuracy=accuracy_score(y_test,prd_value)        
            report[list(models.keys())[i]]=accuracy

     

        return report

    except Exception as e :
        raise CustomException(e,sys)
    



def load_objects(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e :
        raise CustomException (e,sys)
        
    