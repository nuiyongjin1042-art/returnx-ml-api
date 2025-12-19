from fastapi import FastAPI, HTTPException
import os
from supabase import create_client
from predict_pipeline import predict

# ==========================================
# SUPABASE CONFIG
# ==========================================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase environment variables not set")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "ML API running"}


@app.post("/predict")
def run_prediction(payload: dict):
    """
    Receives order JSON, predicts fraud,
    and updates fraud_label in Supabase.
    """
    try:
        order_id = payload.get("order_id")

        if not order_id:
            raise ValueError("order_id is required")

        fraud_label = predict(payload)

        supabase.table("return1_data") \
            .update({"fraud_label": fraud_label}) \
            .eq("order_id", order_id) \
            .execute()

        return {
            "order_id": order_id,
            "fraud_label": fraud_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
