
from pathlib import Path
import pickle

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "student_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_model.pkl"


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    X = df[["study_hours", "attendance", "previous_marks"]]
    y = df["final_marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Configure MLflow using a local SQLite database
    mlflow.set_tracking_uri(
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_experiment("Student Performance Prediction")

    # Train and track experiment
    with mlflow.start_run(run_name="RandomForest Training"):
        mlflow.log_params({
            "model": "RandomForestRegressor",
            "n_estimators": 100,
            "random_state": 42,
            "test_size": 0.2,
        })

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_metrics({
            "MAE": float(mae),
            "R2_Score": float(r2),
        })

        # Save trained model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        # Log model file as an artifact
        mlflow.log_artifact(str(MODEL_PATH))

        print(f"Model saved to: {MODEL_PATH}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R2 Score: {r2:.2f}")
        print("MLflow parameters and metrics logged successfully.")


if __name__ == "__main__":
    main()
