from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"

app = FastAPI(title="Diabetes Prediction API")


class DiabetesFeatures(BaseModel):
    Glucose: float = Field(..., ge=0, example=85)
    BloodPressure: float = Field(..., ge=0, example=65)
    SkinThickness: float = Field(..., ge=0, example=27)
    Insulin: float = Field(..., ge=0, example=0)
    BMI: float = Field(..., ge=0, example=25.3)
    DiabetesPedigreeFunction: float = Field(..., ge=0, example=0.356)
    Age: int = Field(..., gt=0, example=30)


def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


artifact = load_artifact()
model = artifact["model"]
FEATURE_COLUMNS = artifact["features"]


@app.get("/")
def root():
    return {"message": "Diabetes Prediction API is running"}


@app.post("/predict")
def predict(features: DiabetesFeatures):
    try:
        sample = pd.DataFrame([features.model_dump()], columns=FEATURE_COLUMNS)
        prediction = int(model.predict(sample)[0])
        label = "Diabetic" if prediction == 1 else "Non-diabetic"
        return {"prediction": prediction, "label": label}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
