from fastapi import FastAPI, Request, HTTPException
import os
from supabase import create_client, Client
from predict_pipeline import predict

app = FastAPI()

# Load env vars safely
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Supabase environment variables not set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/run")
async def run_model(req: Request):
    body = await req.json()

    if "order_id" not in body:
        raise HTTPException(status_code=400, detail="order_id missing")

    order_id = body["order_id"]

    prediction = predict(body)

    # Convert 0/1 → text
    fraud_text = "fraud" if prediction["fraud_prediction"] == 1 else "not fraud"

    supabase.table("return1_data").update(
        {"fraud_label": fraud_text}
    ).eq("order_id", order_id).execute()

    return {
        "order_id": order_id,
        "fraud_label": fraud_text,
        "probability": prediction["fraud_probability"]
    }
