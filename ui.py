import streamlit as st
from src.prediction import predict_price

st.title("House Price Prediction")

inputs = []
labels = [
    "Longitude", "Latitude", "Median Age",
    "Total Rooms", "Total Bedrooms",
    "Population", "Households", "Median Income"
]

for label in labels:
    value = st.number_input(label, value=1.0)
    inputs.append(value)

if st.button("Predict"):
    result = predict_price(inputs)
    st.success(f"Predicted House Price: ${result:,.2f}")
