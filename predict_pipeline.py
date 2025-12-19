import numpy as np
import pandas as pd
import pickle
import json
import os
from supabase import create_client

# ==========================================
# SUPABASE CONFIG
# ==========================================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "ml-models")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

TMP_DIR = "/tmp"


def load_from_storage(filename, binary=True):
    """
    Download a file from Supabase Storage into /tmp (only once),
    then return the local file path.
    """
    local_path = os.path.join(TMP_DIR, filename)

    if not os.path.exists(local_path):
        print(f"Downloading {filename} from Supabase Storage...")
        data = supabase.storage.from_(MODEL_BUCKET).download(filename)

        mode = "wb" if binary else "w"
        with open(local_path, mode) as f:
            f.write(data)

    return local_path


# ==========================================
# LOAD MODEL ARTIFACTS (HYBRID MODE)
# ==========================================

with open(load_from_storage("xgb_final_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(load_from_storage("best_threshold.json", binary=False), "r") as f:
    threshold = json.load(f)["best_threshold"]

with open(load_from_storage("label_encoders.pkl"), "rb") as f:
    le_map = pickle.load(f)

with open(load_from_storage("imputer.pkl"), "rb") as f:
    imputer = pickle.load(f)

with open(load_from_storage("feature_order.json", binary=False), "r") as f:
    feature_order = json.load(f)

with open(load_from_storage("price_bins.json", binary=False), "r") as f:
    price_bins_by_category = json.load(f)


# ==========================================
# MODEL COLUMNS
# ==========================================

CATEGORICAL_COLUMNS = [
    'category', 'payment_method', 'region', 'customer_gender',
    'age_group', 'discount_group', 'price_group'
]


# ==========================================
# GROUPING FUNCTIONS
# ==========================================

age_bins = [-np.inf, 24, 37, 50, 63, np.inf]
age_labels = ["<25", "25-37", "38-50", "51-63", "64+"]


def get_age_group(age):
    return str(pd.cut(
        [age],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True
    )[0])


def get_discount_group(discount):
    if discount <= 0.10:
        return "0-0.10"
    elif discount <= 0.20:
        return "0.10-0.20"
    else:
        return "0.20-0.30"


def get_price_group(price, category):
    info = price_bins_by_category.get(category)

    if info is None:
        return "Unknown"

    if info["type"] == "single_value":
        return "Low"

    bins = info["bins"]
    labels = ["Low", "Low", "Medium", "High", "Very High", "Very High"]

    group = pd.cut([price], bins=bins, labels=labels, include_lowest=True)
    return str(group[0])


# ==========================================
# PREPROCESSING
# ==========================================

def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    df = df[
        [
            'category', 'price', 'discount', 'quantity', 'payment_method',
            'region', 'total_amount', 'customer_age',
            'customer_gender', 'days_to_return'
        ]
    ]

    df["age_group"] = df["customer_age"].apply(get_age_group)
    df["discount_group"] = df["discount"].apply(get_discount_group)
    df["price_group"] = df.apply(
        lambda row: get_price_group(row["price"], row["category"]),
        axis=1
    )

    for col in CATEGORICAL_COLUMNS:
        df[col] = le_map[col].transform(df[col].astype(str))

    df = pd.DataFrame(imputer.transform(df), columns=df.columns)
    df = df[feature_order]

    return df


# ==========================================
# PREDICTION (FINAL OUTPUT)
# ==========================================

def predict(data: dict):
    """
    Returns fraud label as TEXT:
    - "fraud"
    - "not fraud"
    """
    X = preprocess_input(data)

    prob = model.predict_proba(X)[:, 1][0]
    pred = int(prob >= threshold)

    return "fraud" if pred == 1 else "not fraud"
