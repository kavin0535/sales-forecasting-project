 
import joblib
import pandas as pd
import re
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# =========================
# LOAD FILES (STRICT)

try:
    model = joblib.load('model.joblib')
    store_df = pd.read_csv('store.csv')
    metadata = joblib.load('metadata.joblib')  # store preprocessing info
except Exception as e:
    raise Exception(f"Error loading required files: {e}")

# Extract metadata
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

    # Merge store data
    df = df.merge(store_df, on='Store', how='left')

    # Time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    df.drop(columns=['Date'], inplace=True)

    # Fill missing values
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(median_competition_distance)
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)

    if 'Open' in df.columns:
        df['Open'] = df['Open'].fillna(1)

    df['PromoInterval'] = df['PromoInterval'].fillna('0')
    df['StateHoliday'] = df['StateHoliday'].astype(str).replace({'a': '0', 'b': '0', 'c': '0'})

    # One-hot encoding
    df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

    # Clean column names
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]

    # Align features
    df = df.reindex(columns=model_expected_features, fill_value=0)

    return df


# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return "Sales Forecast API Running"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Empty input"}), 400

        # Support single + batch
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        # Required fields check
        required_cols = ['Store', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        processed = preprocess_input(df)

        preds = model.predict(processed)
        preds = np.maximum(preds, 0)  # no negative sales

        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)


# Write the Flask application code to a file
with open('flask_app.py', 'w') as f:
    f.write(flask_app_code)

print("\nFlask application code exported to 'flask_app.py'.")
print("To run the Flask API locally:")
print("1. Ensure 'model.joblib', 'store.csv', and 'metadata.joblib' are in the same directory as 'flask_app.py'.")
print("2. Open a terminal in that directory.")
print("3. Run the command: `python flask_app.py`")
print("The API will be available at `http://127.0.0.1:5000/predict` and accepts POST requests with JSON data.")
print("\nExample POST request body (single prediction):")
print("""
{
    "Store": 1,
    "DayOfWeek": 5,
    "Date": "2015-07-31",
    "Open": 1.0,
    "Promo": 1,
    "StateHoliday": "0",
    "SchoolHoliday": 1
}
""")
print("\nExample POST request body (batch prediction):")
print("""
[
    {
        "Store": 1,
        "DayOfWeek": 5,
        "Date": "2015-07-31",
        "Open": 1.0,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 1
    },
    {
        "Store": 2,
        "DayOfWeek": 5,
        "Date": "2015-07-31",
        "Open": 1.0,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 1
    }
]
""")
