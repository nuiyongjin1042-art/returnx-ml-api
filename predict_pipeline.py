import numpy as np
import pandas as pd
import pickle
import json

# ==========================================
# LOAD ALL REQUIRED ARTIFACTS
# ==========================================

# 1. Model
with open("xgb_final_model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Threshold
with open("best_threshold.json", "r") as f:
    threshold = json.load(f)["best_threshold"]

# 3. Label Encoders
with open("label_encoders.pkl", "rb") as f:
    le_map = pickle.load(f)

# 4. Imputer
with open("imputer.pkl", "rb") as f:
    imputer = pickle.load(f)

# 5. Feature order (MUST match your training)
with open("feature_order.json", "r") as f:
    feature_order = json.load(f)

# 6. Price bins per category (learned during training)
with open("price_bins.json", "r") as f:
    price_bins_by_category = json.load(f)


# ==========================================
# REQUIRED COLUMNS AS SPECIFIED BY YOU
# ==========================================

MODEL_COLUMNS = [
    'category', 'price', 'discount', 'quantity', 'payment_method',
    'region', 'total_amount', 'customer_age',
    'customer_gender', 'days_to_return',
    'age_group', 'discount_group', 'price_group'
]

CATEGORICAL_COLUMNS = [
    'category', 'payment_method', 'region', 'customer_gender',
    'age_group', 'discount_group', 'price_group'
]


# ==========================================
# GROUPING FUNCTIONS
# ==========================================

# AGE GROUP
age_bins = [-np.inf, 24, 37, 50, 63, np.inf]
age_labels = ["<25", "25-37", "38-50", "51-63", "64+"]


def get_age_group(age):
    return str(pd.cut(
        [age],
        bins=age_bins,
        labels=age_labels,
        right=True,
        include_lowest=True
    )[0])


# DISCOUNT GROUP
def get_discount_group(discount):
    if discount <= 0.10:
        return "0-0.10"
    elif discount <= 0.20:
        return "0.10-0.20"
    else:
        # ALL discounts >0.20 go into the highest bin (model expects 3 bins only)
        return "0.20-0.30"


# PRICE GROUP (using saved bins)
def get_price_group(price, category):
    info = price_bins_by_category.get(category)

    if info is None:
        return "Unknown"

    group_type = info["type"]

    if group_type == "single_value":
        return "Low"

    bins = info["bins"]
    labels = ["Low", "Low", "Medium", "High", "Very High", "Very High"]

    group = pd.cut([price], bins=bins, labels=labels, include_lowest=True, ordered=False )
    return str(group[0])


# ==========================================
# MAIN PREPROCESSING FUNCTION
# ==========================================
def preprocess_input(data: dict):
    """
    Takes raw website JSON and outputs a correctly formatted dataframe
    ready for the trained model.
    """
    df = pd.DataFrame([data])

    # Keep only required columns
    df = df[[
        'category', 'price', 'discount', 'quantity', 'payment_method',
        'region', 'total_amount', 'customer_age',
        'customer_gender', 'days_to_return'
    ]]

    # Add grouped features
    df["age_group"] = df["customer_age"].apply(get_age_group)
    df["discount_group"] = df["discount"].apply(get_discount_group)
    df["price_group"] = df.apply(
        lambda row: get_price_group(row["price"], row["category"]),
        axis=1
    )

    # Label encoding (only these columns)
    for col in CATEGORICAL_COLUMNS:
        df[col] = le_map[col].transform(df[col].astype(str))

    # Impute missing values
    df = pd.DataFrame(imputer.transform(df), columns=df.columns)

    # Reorder columns to match training
    df = df[feature_order]

    return df


# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict(data: dict):
    """
    Returns fraud probability + prediction for a single order.
    """
    X = preprocess_input(data)

    prob = model.predict_proba(X)[:, 1][0]
    pred = int(prob >= threshold)

    return {
        "fraud_probability": float(prob),
        "fraud_prediction": pred
    }