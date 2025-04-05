# Mutual Fund Risk & Safety Score Predictor 🚀

This FastAPI app predicts:
- 📊 Safety Score (Regression)
- ⚠️ Risk Type (Classification)

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API testing.

## Input Format (POST /predict/)

```json
{
  "fund_rating": 4.0,
  "return_1yr": 12.0,
  "return_3yr": 18.5,
  "return_5yr": 22.3,
  "category": "Equity",
  "risk_type": "Moderately High Risk"
}
```
