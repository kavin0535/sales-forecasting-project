 import os
import joblib
import pandas as pd
import numpy as np
import re
from flask import Flask, request, jsonify

app = Flask(__name__)

# =========================
# LOAD FILES (SAFE LOAD)
# =========================
def load_files():
    try:
        model = joblib.load('model.joblib')
        store_df = pd.read_csv('store.csv')
        metadata = joblib.load('metadata.joblib')

        return model, store_df, metadata

    except Exception as e:
        print("🔥 FILE LOADING ERROR:", e)
        raise e

model, store_df, metadata = load_files()

median_competition_distance = metadata['median_competition_distance']
model_expected_features = metadata['model_expected_features']
categorical_cols = metadata['categorical_cols']


# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_input(df):
    if 'Date' not in df.columns:
        raise ValueError("Missing 'Date' column")

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.merge(store_df, on='Store', how='left')

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    df.drop(columns=['Date'], inplace=True)

    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(median_competition_distance)
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)

    if 'Open' in df.columns:
        df['Open'] = df['Open'].fillna(1)

    df['PromoInterval'] = df['PromoInterval'].fillna('0')
    df['StateHoliday'] = df['StateHoliday'].astype(str).replace({'a': '0', 'b': '0', 'c': '0'})

    df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]

    df = df.reindex(columns=model_expected_features, fill_value=0)

    return df


# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return "API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Empty input"}), 400

        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        required_cols = ['Store', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        processed = preprocess_input(df)

        preds = model.predict(processed)
        preds = np.maximum(preds, 0)

        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        print("🔥 PREDICTION ERROR:", e)
        return jsonify({"error": str(e)}), 500


# =========================
# ENTRY POINT (RENDER SAFE)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
