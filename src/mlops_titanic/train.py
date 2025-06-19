import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import mlflow
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("TitanicModels")

try:
    # При запуске через Snakemake
    processed_input = snakemake.input.train_p
    output_paths = snakemake.output.models
except NameError:
    # При ручном запуске
    processed_input = "data/processed/train_preprocessed.csv"
    output_paths = [f"models/{name}.pkl" for name in [
        "logistic_regression", "knn", "svc", "naive_bayes", "decision_tree", "random_forest", "perceptron"
    ]]

Path(output_paths[0]).parent.mkdir(parents=True, exist_ok=True)

def load_processed():
    df = pd.read_csv(processed_input)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return X, y

MODELS = {
    "logistic_regression": LogisticRegression(C=10.0, solver='liblinear', max_iter=300, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=3, weights='distance'),
    "svc": SVC(C=1.0, kernel='poly', degree=3, probability=True, random_state=42),
    "naive_bayes": GaussianNB(var_smoothing=1e-9),
    "decision_tree": DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=4, random_state=42),
    "perceptron": Perceptron(penalty='elasticnet', alpha=0.001, l1_ratio=0.15, random_state=42),
}

if __name__ == "__main__":
    X, y = load_processed()
    for model_name, model in MODELS.items():
        path = Path(f"models/{model_name}.pkl")
        if path in [Path(p) for p in output_paths]:
            with mlflow.start_run(run_name=f"{model_name}_exp3"):
                # Логируем параметры
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("model_params", str(model.get_params()))
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("n_samples", X.shape[0])

                # Обучение
                model.fit(X, y)
                joblib.dump(model, path)

                # Логируем точность на трейне
                acc = model.score(X, y)
                mlflow.log_metric("train_accuracy", acc)

                # Логируем модель как артефакт
                mlflow.log_artifact(path)

                print(f"Saved and logged: {model_name}")