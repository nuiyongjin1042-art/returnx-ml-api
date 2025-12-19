import os
import json
import pickle
import requests
import numpy as np
import pandas as pd

# ==========================
# CONFIG
# ==========================

SUPABASE_PROJECT_ID = os.environ["SUPABASE_PROJECT_ID"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]

BUCKET = "ml-models"
BASE_URL = f"https://{SUPABASE_PROJECT_ID}.supabase.co/storage/v1/object/public/{BUCKET}"

TMP_DIR = "/tmp"

FILES = {
    "model": "xgb_final_model.pkl",
    "threshold": "best_threshold.json",
    "encoders": "label_encoders.pkl",
    "imputer": "imputer.pkl",
    "feature_order": "feature_order.json",
    "price_bins": "price_bins.json",
}

# ==========================
# STORAGE DOWNLOAD HELPER
# ==========================

def load_from_storage(filename, binary=True):
    local_path = f"{TMP_DIR}/{filename}"

    if not os.path.exists(local_path):
        url = f"{BASE_URL}/{filename}"
        r = requests.get(url)
        r.raise_for_status()

        mode = "wb" if binary else "w"
        with open(local_path, mode) as f:
            f.write(r.content if binary else r.text)

    return local_path

# ==========================
# LOAD ARTIFACTS
# ==========================

with open(load_from_storage(FILES["model"]), "rb") as f:
    model = pickle.load(f)

with open(load_from_storage(FILES["threshold"], binary=False), "r") as f:
    threshold = json.load(f)["best_threshold"]

with open(load_from_storage(FILES["encoders"]), "rb") as f:
    le_map = pickle.load(f)

with open(load_from_storage(FILES["imputer"]), "rb") as f:
    imputer = pickle.load(f)

with open(load_from_storage(FILES["feature_order"], binary=False), "r") as f:
    feature_order = json.load(f)

with open(load_from_storage(FILES["price_bins"], binary=False), "r") as f:
    price_bins_by_category = json.load(f)

# ==========================
# FEATURE ENGINEERING
# ==========================

CATEGORICAL_COLUMNS = [
    "category", "payment_method", "region", "customer_gender",
    "age_group", "discount_group", "price_group"
]

age_bins = [-np.inf, 24, 37, 50, 63, np.inf]
age_labels = ["<25", "25-37", "38-50", "51-63", "64+"]

def get_age_group(age):
    return str(pd.cut([age], bins=age_bins, labels=age_labels)[0])

def get_discount_group(discount):
    if discount <= 0.10:
        return "0-0.10"
    elif discount <= 0.20:
        return "0.10-0.20"
    return "0.20-0.30"

def get_price_group(price, category):
    info = price_bins_by_category.get(category)
    if not info:
        return "Unknown"

    if info["type"] == "single_value":
        return "Low"

    bins = info["bins"]
    labels = ["Low", "Low", "Medium", "High", "Very High", "Very High"]
    return str(pd.cut([price], bins=bins, labels=labels)[0])

# ==========================
# PREDICTION
# ==========================

def predict(data: dict):
    df = pd.DataFrame([data])

    df["age_group"] = df["customer_age"].apply(get_age_group)
    df["discount_group"] = df["discount"].apply(get_discount_group)
    df["price_group"] = df.apply(
        lambda r: get_price_group(r["price"], r["category"]), axis=1
    )

    for col in CATEGORICAL_COLUMNS:
        df[col] = le_map[col].transform(df[col].astype(str))

    df = pd.DataFrame(imputer.transform(df), columns=df.columns)
    df = df[feature_order]

    prob = model.predict_proba(df)[:, 1][0]
    pred = "fraud" if prob >= threshold else "not fraud"

    return {"fraud_label": pred}
