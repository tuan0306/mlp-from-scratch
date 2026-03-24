from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import sys
import pickle
sys.path.append('../')
from core.utils import prob_to_class,preprocess_user_input

app=FastAPI(
    title='Titanic AI Pricdiction API',
    description='Microservice dự đoán khả năng sống sót trên tàu Titanic bằng Mạng Nơ-ron NumPy',
    version='1.0.0'
)

class PassengerInput(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    SibSp: int
    Parch: int
    Embarked: str
    
try:
    with open('models/titanic_best_model.pkl','rb') as f:
        ai_model=pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler=pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Can not load Model file or Scaler file. Details: {e}")

@app.post('/predict')
def predict_survival(passenger: PassengerInput):
    try:
        raw_data=passenger.dict()
        final_features=preprocess_user_input(raw_data,scaler)
        raw_prediction=ai_model.predict(final_features)
        survival_prob=raw_prediction.item()
        
        survival_class=1 if survival_prob >= 0.5 else 0
        return {
            'status':'succes',
            'prediction_class':survival_class,
            'prediction_probability':round(survival_prob,4),
            'prediction_label':'Survived' if survival_class==1 else 'Dead'
        }
    except Exception as e:
        raise HTTPException(status_code=5000,detail=f"Predicting error: {str(e)}")