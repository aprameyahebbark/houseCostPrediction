from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model (pipeline included)
model = joblib.load("model.pkl")

def format_price(price):
    if price >= 1e7:
        return f"₹ {price/1e7:.2f} Cr"
    elif price >= 1e5:
        return f"₹ {price/1e5:.2f} Lakh"
    else:
        return f"₹ {int(price):,}"

@app.route('/')
def home():
    return render_template("index.html", prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract + validate inputs
        location = request.form['location'].strip().lower()
        area = float(request.form['Carpet Area'])
        floor = int(request.form['Floor'])
        bathroom = int(request.form['Bathroom'])
        balcony = int(request.form['Balcony'])
        parking = int(request.form['Car Parking'])

        # Validation (backend safety)
        if not (300 <= area <= 3000):
            raise ValueError("Carpet Area must be 300–3000")
        if not (0 <= floor <= 30):
            raise ValueError("Floor must be 0–30")
        if not (1 <= bathroom <= 6):
            raise ValueError("Bathroom must be 1–6")
        if not (0 <= balcony <= 5):
            raise ValueError("Balcony must be 0–5")

        data = {
            "location": location,
            "Carpet Area": area,
            "Status": request.form['Status'],
            "Floor": floor,
            "Transaction": request.form['Transaction'],
            "Furnishing": request.form['Furnishing'],
            "Bathroom": bathroom,
            "Balcony": balcony,
            "Car Parking": parking
        }

        df = pd.DataFrame([data])

        # Predict (log → actual)
        log_price = model.predict(df)[0]
        price = np.exp(log_price)

        return render_template(
            "index.html",
            prediction_text=f"Estimated Price: {format_price(price)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)