import joblib
import pandas as pd


def main():
    # Load trained model and label encoder
    model = joblib.load("food_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Change these values to predict a new day
    new_data = pd.DataFrame([
        {
            "day_of_week": "Monday",
            "month": 4,
            "weather": "Rainy",
            "temperature": 8,
            "yesterday_food": "takeaway",
        }
    ])

    # Predict class
    pred_encoded = model.predict(new_data)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    print("=" * 50)
    print("TODAY'S FOOD PREDICTION")
    print("=" * 50)
    print(f"Predicted food: {pred_label}")

    # Predict probabilities
    probabilities = model.predict_proba(new_data)[0]
    class_names = label_encoder.inverse_transform(range(len(probabilities)))

    ranked = sorted(
        zip(class_names, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nProbabilities:")
    for food, prob in ranked:
        print(f"{food}: {prob:.2%}")


if __name__ == "__main__":
    main()