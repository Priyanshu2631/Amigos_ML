from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# === TRAINING AT STARTUP ===
df = pd.read_csv("mutual_funds_india.csv")

df.drop(columns=["Unnamed: 0", "AMC_name", "Mutual Fund Name"], inplace=True, errors="ignore")
imputer = SimpleImputer(strategy='median')
df[["fund_rating", "return_1yr", "return_3yr", "return_5yr"]] = imputer.fit_transform(
    df[["fund_rating", "return_1yr", "return_3yr", "return_5yr"]]
)
scaler = MinMaxScaler()
df[["return_1yr", "return_3yr", "return_5yr"]] = scaler.fit_transform(
    df[["return_1yr", "return_3yr", "return_5yr"]]
)

le_category = LabelEncoder()
le_risk = LabelEncoder()
df["category"] = le_category.fit_transform(df["category"])
df["risk_type_encoded"] = le_risk.fit_transform(df["risk_type"])

risk_mapping = {
    'Low Risk': 5, 'Moderately Low Risk': 4, 'Moderate Risk': 3,
    'Moderately High Risk': 2, 'Very High Risk': 1
}
df["risk_score"] = df["risk_type"].map(risk_mapping)
df.drop(columns=["risk_type"], inplace=True)

df["safety_score"] = (df["fund_rating"] * 2 + df["return_1yr"] + df["return_3yr"] + df["return_5yr"] + df["risk_score"]) / 6
df = df.dropna(subset=["safety_score"])

# Train Regression Model
y_reg = df["safety_score"]
X_reg = df.drop(columns=["safety_score", "risk_type_encoded", "category"])
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_reg, y_reg)

# Train Classification Model
y_cls = df["risk_type_encoded"]
X_cls = df.drop(columns=["risk_type_encoded", "safety_score"])
scaler_cls = StandardScaler()
X_cls_scaled = scaler_cls.fit_transform(X_cls)
cls_model = RandomForestClassifier(n_estimators=250, random_state=42)
cls_model.fit(X_cls_scaled, y_cls)

# === FASTAPI SETUP ===
app = FastAPI()

class FundInput(BaseModel):
    fund_rating: float
    return_1yr: float
    return_3yr: float
    return_5yr: float
    category: str
    risk_type: str

@app.post("/predict/")
def predict(data: FundInput):
    try:
        input_data = {
            "category": le_category.transform([data.category])[0],
            "risk_type_encoded": le_risk.transform([data.risk_type])[0],
            "risk_score": risk_mapping[data.risk_type],
            "fund_rating": data.fund_rating,
            "return_1yr": data.return_1yr,
            "return_3yr": data.return_3yr,
            "return_5yr": data.return_5yr,
        }

        df_input = pd.DataFrame([input_data])
        df_input[["return_1yr", "return_3yr", "return_5yr"]] = scaler.transform(
            df_input[["return_1yr", "return_3yr", "return_5yr"]]
        )

        reg_pred = reg_model.predict(df_input[X_reg.columns])[0]
        cls_pred = cls_model.predict(scaler_cls.transform(df_input[X_cls.columns]))[0]
        risk_label = le_risk.inverse_transform([cls_pred])[0]

        return {
            "safety_score": round(float(reg_pred), 4),
            "predicted_risk_type": risk_label
        }

    except Exception as e:
        return {"error": str(e)}
