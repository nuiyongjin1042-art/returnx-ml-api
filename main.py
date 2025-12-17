from fastapi import FastAPI, Request, HTTPException
import os
from supabase import create_client, Client
from predict_pipeline import predict_pipeline

# Load Supabase credentials from Cloud Run env vars
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Supabase environment variables not set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI()

@app.post("/run")
async def run_model(req: Request):
    body = await req.json()

    # Validate required field
    if "order_id" not in body:
        raise HTTPException(status_code=400, detail="order_id missing")

    order_id = body["order_id"]

    try:
        # Run ML model
        prediction = predict_pipeline(body)

        # Update Supabase
        supabase.table("return1_data").update(
            {"fraud_label": prediction}
        ).eq("order_id", order_id).execute()

        return {
            "status": "ok",
            "order_id": order_id,
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
