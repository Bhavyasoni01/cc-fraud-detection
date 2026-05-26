from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI();

with open('saved_model.pkl','rb') as f:
    model = joblib.load(f)

class Transaction(BaseModel):
    Time:float
    V1:float
    V2:float
    V3:float
    V4:float
    V5:float
    V6:float
    V7:float
    V8:float
    V9:float
    V10:float
    V11:float
    V12:float
    V13:float
    V14:float
    V15:float
    V16:float
    V17:float
    V18:float
    V19:float
    V20:float
    V21:float
    V22:float
    V23:float
    V24:float
    V25:float
    V26:float
    V27:float
    V28:float
    Amount:float


@app.get("/")
def home():
    return{
        "message" : "Credit Card Fraud Detection Model"
    }

@app.post("/predict")
def predict(transaction : Transaction):
    data = pd.DataFrame([transaction.model_dump()])

    probabilty = float((model.predict_proba(data)[0][1]))
    predictions = "Fraud" if probabilty >= 0.7 else "Legit"

    return {
        'prediction' : predictions,
        'fraud_probability_percent' : round(probabilty * 100,2)
    }


