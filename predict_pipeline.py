import numpy as np
import pandas as pd
import pickle
import json

# ===============================
# LOAD ALL MODEL ARTIFACTS (LOCAL)
# ===============================

with open("xgb_final_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("best_threshold.json", "r") as f:
    threshold = json.load(f)["best_threshold"]

with open("label_encoders.pkl", "rb") as f:
    le_map = pickle.load(f)

with open("imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

with open("feature_order.json", "r") as f:
    feature_order = json.load(f)

with open("price_bins.json", "r") as f:
    price_bins_by_category = json.load(f)

# ===============================
# CONFIG
# ===============================

CATEGORICAL_COLUMNS = [
    "category",
    "payment_method",
    "region",
    "customer_gender",
    "age_group",
    "discount_group",
    "price_group"
]

# ===============================
# GROUPING FUNCTIONS
# ===============================

age_bins = [-np.inf, 24, 37, 50, 63, np.inf]
age_labels = ["<25", "25-37", "38-50", "51-63", "64+"]

def get_age_group(age):
    return str(pd.cut([age], bins=age_bins, labels=age_labels)[0])

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

    group_type = info["type"]

    if group_type == "single_value":
        return "Low"

    bins = info["bins"]
    labels = ["Low", "Low", "Medium", "High", "Very High", "Very High"]

    group = pd.cut(
        [price],
        bins=bins,
        labels=labels,
        include_lowest=True,
        ordered=False   # 🔥 THIS IS THE FIX
    )

    return str(group[0])

# ===============================
# PREPROCESS
# ===============================

def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    df = df[
        [
            "category",
            "price",
            "discount",
            "quantity",
            "payment_method",
            "region",
            "total_amount",
            "customer_age",
            "customer_gender",
            "days_to_return",
        ]
    ]

    df["age_group"] = df["customer_age"].apply(get_age_group)
    df["discount_group"] = df["discount"].apply(get_discount_group)
    df["price_group"] = df.apply(
        lambda row: get_price_group(row["price"], row["category"]), axis=1
    )

    for col in CATEGORICAL_COLUMNS:
        df[col] = le_map[col].transform(df[col].astype(str))

    df = pd.DataFrame(imputer.transform(df), columns=df.columns)
    df = df[feature_order]

    return df

# ===============================
# PREDICT
# ===============================

def predict(data: dict):
    X = preprocess_input(data)
    prob = model.predict_proba(X)[:, 1][0]
    pred = int(prob >= threshold)

    return {
        "fraud_prediction": "fraud" if pred == 1 else "not fraud"
    }

