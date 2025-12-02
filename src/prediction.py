import joblib
import numpy as np

def predict_price(input_features):
    # Load the model
    model = joblib.load("model.pkl")
    
    # Convert features to numpy array
    input_array = np.array(input_features).reshape(1, -1)
    
    # Predict
    prediction = model.predict(input_array)[0]

    # Prevent negative price output
    prediction = max(0, prediction)

    return prediction
