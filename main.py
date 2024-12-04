from fastapi import FastAPI, Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse
import joblib
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
model = joblib.load('model.pkl')
app = FastAPI()
security = HTTPBearer()

API_KEY = os.getenv("API_KEY")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.get("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float, credentials: HTTPAuthorizationCredentials = Depends(verify_api_key)):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return {"prediction": species_map[int(prediction[0])]}
