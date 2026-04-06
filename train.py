import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def main():
    # Load data
    df = pd.read_csv("data/office_food.csv")

    # Optional safety check
    required_columns = [
        "day_of_week",
        "month",
        "weather",
        "temperature",
        "yesterday_food",
        "food_category",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Features and target
    X = df[["day_of_week", "month", "weather", "temperature", "yesterday_food"]]
    y = df["food_category"]

    # Encode target labels separately
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Columns
    categorical_features = ["day_of_week", "weather", "yesterday_food"]
    numerical_features = ["month", "temperature"]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    # Model pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=8,
                    random_state=42
                ),
            ),
        ]
    )

    print(df["food_category"].value_counts())
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded,
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    k = min(3, len(label_encoder.classes_))
    top3_accuracy = top_k_accuracy_score(
        y_test,
        y_proba,
        k=k,
        labels=list(range(len(label_encoder.classes_)))
    )

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Top-{k} Accuracy: {top3_accuracy:.2f}")
    print("\nClassification Report:\n")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(len(label_encoder.classes_))),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    # Save trained model + label encoder
    joblib.dump(model, "food_model.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")

    print("Saved model to food_model.pkl")
    print("Saved label encoder to label_encoder.pkl")


if __name__ == "__main__":
    main()