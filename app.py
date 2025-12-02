from src.utils import load_data
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.prediction import predict_price

def main():
    print("Loading dataset...")
    df = load_data("data/housing.csv")

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    print("Training model...")
    mse = train_model(X_train, y_train, X_test, y_test)
    print(f"Model trained successfully! MSE: {mse}")

    print("\nMaking a sample prediction...")
    sample = X_test[0]
    prediction = predict_price(sample)
    print(f"Predicted price: {prediction}")

if __name__ == "__main__":
    main()
