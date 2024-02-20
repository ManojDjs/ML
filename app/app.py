from fastapi import FastAPI
import joblib
import pandas as pd
# Third party imports
from pydantic import BaseModel, Field
app = FastAPI()
model = joblib.load('finalized_model.sav')


# Input for data validation
class Input(BaseModel):
    Avg_Area_Income: float = Field(..., gt=0)
    Avg_Area_House_Age: float = Field(..., gt=0)
    Avg_Area_Number_of_Rooms: float = Field(..., gt=0)
    Avg_Area_Number_of_Bedrooms: float = Field(..., gt=0)
    Area_Population: float = Field(..., gt=0)

    class Config:
        schema_extra = {
            "Avg_Area_Income": 0.3001,
            "Avg_Area_House_Age": 0.1471,
            "Avg_Area_Number_of_Rooms": 8.589,
            "Avg_Area_Number_of_Bedrooms": 153.4,
            "Area_Population": 2019.0,
        }



def predict(X, model):
    prediction = model.predict(X)
    return prediction


def get_model_response(input):
    X = pd.json_normalize(input.__dict__)
    print(X)
    prediction = predict(X, model)
    if prediction == 1:
        label = "M"
    else:
        label = "B"
    return {
        'label': label,
        'prediction': int(prediction)
    }
@app.get("/")
async def root():
    return {"message": "Hello World"}


# Ouput for data validation
class Output(BaseModel):
    price: float


@app.post('/predict', response_model=Output)
async def model_predict(input: Input):
    """Predict with input"""
    response = get_model_response(input)
    return response