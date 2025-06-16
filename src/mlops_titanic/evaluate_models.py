import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score

models = snakemake.input.models
test_processed = snakemake.input.test_processed
target_path = snakemake.input.target
output_path = snakemake.output[0]

def load_test_data():
    X_test = pd.read_csv(test_processed)
    y_true = pd.read_csv(target_path)["Survived"]
    return X_test, y_true

def evaluate_model(model_path: Path, X_test: pd.DataFrame, y_true: pd.Series):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    return accuracy_score(y_true, y_pred)

if __name__ == "__main__":
    results = []
    for model_path in models:
        name = Path(model_path).stem
        try:
            X_test, y_true = load_test_data()
            acc = evaluate_model(model_path, X_test, y_true)
            results.append((name, acc))
        except Exception as e:
            print(f"Failed to evaluate {name}: {e}")

    results.sort(key=lambda x: x[1], reverse=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Accuracy ranking:\n")
        for name, acc in results:
            f.write(f"{name:<30} accuracy: {acc:.4f}\n")