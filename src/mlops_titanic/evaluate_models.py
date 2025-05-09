import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score

DATA_PATH = Path("data")
MODELS_PATH = Path("models")
TARGET_PATH = DATA_PATH / "raw" / "gender_submission.csv"


def load_test_data(dataset_type: str):
    if dataset_type == "processed":
        X_test = pd.read_csv(DATA_PATH / "processed" / "test_preprocessed.csv")
    elif dataset_type == "raw":
        df = pd.read_csv(DATA_PATH / "raw" / "test.csv")
        df = df[["Pclass", "Sex", "SibSp", "Parch", "Fare"]].copy()
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
        X_test = df.fillna(df.median(numeric_only=True))
    else:
        raise ValueError("Unknown dataset type")
    
    y_true = pd.read_csv(TARGET_PATH)["Survived"]
    return X_test, y_true


def evaluate_model(model_path: Path, X_test: pd.DataFrame, y_true: pd.Series):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    return accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    results = []

    for model_file in MODELS_PATH.glob("*.pkl"):
        model_name = model_file.stem
        if model_name.endswith("processed"):
            dataset_type = "processed"
        elif model_name.endswith("raw"):
            dataset_type = "raw"
        else:
            print(f"Skipping unknown model type: {model_name}")
            continue

        try:
            X_test, y_true = load_test_data(dataset_type)
            acc = evaluate_model(model_file, X_test, y_true)
            results.append((model_name, acc))
        except Exception as e:
            print(f"Failed to evaluate {model_name}: {e}")

    # Сортировка по убыванию точности
    results.sort(key=lambda x: x[1], reverse=True)

    print("\nAccuracy ranking:")
    for name, acc in results:
        print(f"{name:<30} accuracy: {acc:.4f}")
