from fastapi import FastAPI, HTTPException
import os, json, joblib
from supabase import create_client
from predict_pipeline import predict

# ----------------------------
# 1. Environment variables
# ----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
MODEL_BUCKET = "ml-models"

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing Supabase env vars")

# ----------------------------
# 2. Supabase client (service role)
# ----------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ----------------------------
# 3. Download model files ONCE
# ----------------------------
def download_model_files():
    files = [
        "xgb_final_model.pkl",
        "imputer.pkl",
        "label_encoders.pkl",
        "feature_order.json",
        "price_bins.json",
        "best_threshold.json"
    ]

    for f in files:
        print(f"Downloading {f} from Supabase Storage...")
        data = supabase.storage.from_(MODEL_BUCKET).download(f)
        with open(f, "wb") as out:
            out.write(data)

download_model_files()

# ----------------------------
# 4. Load model into memory
# ----------------------------
model = joblib.load("xgb_final_model.pkl")
imputer = joblib.load("imputer.pkl")
label_encoders = joblib.load("label_encoders.pkl")

with open("feature_order.json") as f:
    feature_order = json.load(f)

with open("price_bins.json") as f:
    price_bins = json.load(f)

with open("best_threshold.json") as f:
    best_threshold = json.load(f)["best_threshold"]

print("✅ Model loaded and ready")

# ----------------------------
# 5. FastAPI app
# ----------------------------
app = FastAPI()

@app.post("/predict")
def run_prediction(payload: dict):
    try:
        order_id = payload["order_id"]

        result = predict(
            payload,
            model=model,
            imputer=imputer,
            label_encoders=label_encoders,
            feature_order=feature_order,
            price_bins=price_bins,
            threshold=best_threshold
        )

        # Update Supabase
        supabase.table("return1_data") \
            .update({
                "fraud_label": result["fraud_prediction"],
                "fraud_probability": result["fraud_probability"]
            }) \
            .eq("order_id", order_id) \
            .execute()

        return {
            "status": "ok",
            "order_id": order_id,
            **result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
