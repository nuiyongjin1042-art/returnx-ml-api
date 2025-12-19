from fastapi import FastAPI, Request
from predict_pipeline import predict

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ML API running"}


@app.post("/predict")
async def run_prediction(req: Request):
    data = await req.json()
    result = predict(data)
    return result

