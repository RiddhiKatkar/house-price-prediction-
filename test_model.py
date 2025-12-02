from src.prediction import predict_price
import numpy as np

# Example input (replace with real features)
# Based on numeric columns of dataset after dropping non-numeric:
# ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
#  'total_bedrooms', 'population', 'households', 'median_income']

sample_input = [
    -122.23,  # longitude
    37.88,    # latitude
    41,       # age
    880,      # rooms
    129,      # bedrooms
    322,      # population
    126,      # households
    8.33      # income
]

prediction = predict_price(sample_input)
print("Predicted house price:", prediction)
