from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle

# Load models and encoders
with open("models/reg_model.pkl", "rb") as f:
    reg_model = pickle.load(f)
with open("models/cls_model.pkl", "rb") as f:
    cls_model = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/sc.pkl", "rb") as f:
    scaler_cls = pickle.load(f)
with open("models/le_category.pkl", "rb") as f:
    le_category = pickle.load(f)
with open("models/le_risk.pkl", "rb") as f:
    le_risk = pickle.load(f)

# Risk mapping used during training
risk_mapping = {
    'Low Risk': 5, 'Moderately Low Risk': 4, 'Moderate Risk': 3,
    'Moderately High Risk': 2, 'Very High Risk': 1
}

app = FastAPI(title="Mutual Fund Risk API")

class FundInput(BaseModel):
    fund_rating: float
    return_1yr: float
    return_3yr: float
    return_5yr: float
    category: str
    risk_type: str

@app.post("/predict/")
def predict_risk_and_safety(data: FundInput):
    try:
        processed_input = {}

        processed_input['category'] = le_category.transform([data.category])[0]
        processed_input['risk_type_encoded'] = le_risk.transform([data.risk_type])[0]
        processed_input['risk_score'] = risk_mapping[data.risk_type]

        processed_input['fund_rating'] = data.fund_rating
        processed_input['return_1yr'] = data.return_1yr
        processed_input['return_3yr'] = data.return_3yr
        processed_input['return_5yr'] = data.return_5yr

        df_input = pd.DataFrame([processed_input])
        df_input[["return_1yr", "return_3yr", "return_5yr"]] = scaler.transform(
            df_input[["return_1yr", "return_3yr", "return_5yr"]]
        )

        # Predict safety score
        reg_features = df_input[reg_model.feature_names_in_]
        safety_score = float(reg_model.predict(reg_features)[0])

        # Predict risk type
        cls_features = df_input[cls_model.feature_names_in_]
        cls_features_scaled = scaler_cls.transform(cls_features)
        risk_encoded = cls_model.predict(cls_features_scaled)[0]
        risk_label = le_risk.inverse_transform([risk_encoded])[0]

        return {
            "safety_score": round(safety_score, 4),
            "predicted_risk_type": risk_label
        }

    except Exception as e:
        return {"error": str(e)}
