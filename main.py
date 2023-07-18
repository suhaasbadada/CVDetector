from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model.readings_model import preprocess_readings_input_readings,xgboost
from model.lifestyle_model import preprocess_readings_input_lifestyle,run_model
from pydantic import BaseModel

app = FastAPI()

origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=origins,
    allow_headers=origins
)

class ReadingsInput(BaseModel):
    age: int
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    physical_activity: str
    height: float
    weight: int
    gender: str


class LifestyleInput(BaseModel):
    General_Health: str
    Checkup: str
    Exercise: str
    Skin_Cancer: str
    Other_Cancer: str
    Depression: str
    Diabetes: str
    Arthritis: str
    Sex: str
    Height: float
    Weight: int
    Smoking_History: str
    Alcohol_Consumption: int 
    Fruit_Consumption: int 
    Green_Vegetables_Consumption: int
    FriedPotato_Consumption: int
    Age_range: str

@app.post("/readings")
async def predict_CVD_from_readings(input_data: ReadingsInput):
    json=preprocess_readings_input_readings(dict(input_data))
    return {"readings": xgboost(json)}

@app.post("/lifestyle")
async def predict_CVD_from_lifestyle(input_data: LifestyleInput):
    json=preprocess_readings_input_lifestyle(dict(input_data))
    return {"lifestyle": run_model(json)}