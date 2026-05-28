import os 
import sys
import pandas as pd
from src.utils import load_objects
from src.exception import CustomException


class predictionPipeline():
    def __init__(self):
        self.model_path=os.path.join("artifacts","model.pkl")
        self.processor_path = os.path.join("artifacts","processor.pkl")

    try:

        def predict_(self,feature):

            model = load_objects(self.model_path)

            processor = load_objects(self.processor_path)

            processed_data=processor.transform(feature)

            predicted_value = model.predict(processed_data)

            return predicted_value
        
    except Exception as e :
        raise CustomException(e,sys)
    

class CustomData():
    def __init__(self,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,
                 StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges):
        self.gender=gender
        self.SeniorCitizen=SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService=PhoneService
        self.MultipleLines=MultipleLines
        self.InternetService=InternetService
        self.OnlineSecurity=OnlineSecurity
        self.OnlineBackup=OnlineBackup
        self.DeviceProtection=DeviceProtection
        self.TechSupport=TechSupport
        self.StreamingTV=StreamingTV
        self.StreamingMovies=StreamingMovies
        self.Contract=Contract
        self.PaperlessBilling=PaperlessBilling
        self.PaymentMethod=PaymentMethod
        self.MonthlyCharges=MonthlyCharges
        self.TotalCharges=TotalCharges
    try:
        def get_dataframe(self):
            dataframe= {
                "gender" : self.gender,
                "SeniorCitizen" : self.SeniorCitizen,
                "Partner":self.Partner,
                "Dependents":self.Dependents,
                "tenure":self.tenure,
                "PhoneService":self.PhoneService,
                "MultipleLines":self.MultipleLines,
                "InternetService":self.InternetService,
                "OnlineSecurity":self.OnlineSecurity,
                "OnlineBackup":self.OnlineBackup,
                "DeviceProtection":self.DeviceProtection,
                "TechSupport":self.TechSupport,
                "StreamingTV":self.StreamingTV,
                "StreamingMovies":self.StreamingMovies,
                "Contract":self.Contract,
                "PaperlessBilling":self.PaperlessBilling,
                "PaymentMethod": self.PaymentMethod,
                "MonthlyCharges":self.MonthlyCharges,
                "TotalCharges":self.TotalCharges,
            }

            return pd.DataFrame([dataframe])        
    except Exception as e :
        raise CustomException(e,sys)
    
    




