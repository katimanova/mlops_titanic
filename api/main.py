from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import yaml

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].astype(str).map({"S": 0, "C": 1, "Q": 2}).astype(int)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df

class Passenger(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str
    Embarked: str

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running!"}

with open("api/config.yaml") as f:
    config = yaml.safe_load(f)
model = joblib.load(config["model_path"])

@app.post("/predict")
def predict(passenger: Passenger):
    df = pd.DataFrame([passenger.dict()])
    preprocess_df = preprocess(df)
    prediction = model.predict(preprocess_df)[0]
    return {"prediction": int(prediction)}