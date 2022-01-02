import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PBI_user(BaseModel):
    marital: str
    housing: str
    loan: str
    contact: str
    month: str
    duration: int
    campaign: int
    poutcome: str


encoder = joblib.load("dump/cat_encoder.dump")
model = joblib.load("dump/lr_model.dump")
scale = joblib.load("dump/scale.dump")


@app.get("/")
def home():
    return {"Status": "Server is Up and Running"}


@app.post("/predict")
def predict(data: PBI_user):
    data = pd.DataFrame(pd.Series(data)).T
    encoder = joblib.load("dump/cat_encoder.dump")
    model = joblib.load("dump/lr_model.dump")
    scale = joblib.load("dump/scale.dump")
    data = encoder.transform(data)
    data = pd.DataFrame(scale.transform(data), columns=data.columns)
    predicted_val = model.predict(data)
    return {"return": predicted_val}


if __name__ == '__main__':
    uvicorn.run('predict:app', port=8000, reload=True)
