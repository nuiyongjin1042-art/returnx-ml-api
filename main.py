from fastapi import FastAPI
from predict_pipeline import predict

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_api(payload: dict):
    return predict(payload)
