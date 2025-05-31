# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the model
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get form inputs
#     area = float(request.form['area'])
#     bedrooms = int(request.form['bedrooms'])
#     bathrooms = int(request.form['bathrooms'])
#     floors = int(request.form['floors'])
#     garage = int(request.form['garage'])  # 1 for Yes, 0 for No

#     # Make prediction
#     features = np.array([[area, bedrooms, bathrooms, floors, garage]])
#     prediction = model.predict(features)[0]

#     return render_template('index.html', prediction_text=f'Estimated House Price: ₹{prediction:,.2f}')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# Extract unique values from feature_columns for dropdowns
locations = sorted([col.replace("Location_", "") for col in feature_columns if col.startswith("Location_")])
conditions = sorted([col.replace("Condition_", "") for col in feature_columns if col.startswith("Condition_")])
garages = sorted([col.replace("Garage_", "") for col in feature_columns if col.startswith("Garage_")])

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Numeric inputs
            area = float(request.form['area'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            floors = int(request.form['floors'])
            yearbuilt = int(request.form['yearbuilt'])
            
            # Categorical inputs
            location = request.form['location']
            condition = request.form['condition']
            garage = request.form['garage']

            # Prepare input dictionary with numeric values
            input_dict = {
                "Area": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Floors": floors,
                "YearBuilt": yearbuilt,
            }

            # One-hot encode categorical variables
            for col in feature_columns:
                if col.startswith("Location_"):
                    input_dict[col] = 1 if col == f"Location_{location}" else 0
                elif col.startswith("Condition_"):
                    input_dict[col] = 1 if col == f"Condition_{condition}" else 0
                elif col.startswith("Garage_"):
                    input_dict[col] = 1 if col == f"Garage_{garage}" else 0

            # Fill missing columns with 0 if any
            for col in feature_columns:
                if col not in input_dict:
                    input_dict[col] = 0

            # Convert to DataFrame and scale
            input_df = pd.DataFrame([input_dict])[feature_columns]
            input_scaled = scaler.transform(input_df)

            # Predict price
            predicted_price = model.predict(input_scaled)[0]
            prediction = round(predicted_price, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction,
                           locations=locations, conditions=conditions, garages=garages)

if __name__ == "__main__":
    app.run(debug=True)
