import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Пути, полученные из Snakefile
processed_input = snakemake.input.train_p  # data/processed/train_preprocessed.csv
output_paths = snakemake.output.models     # список из 14 моделей (processed+raw)

# Убедимся, что директория существует
Path(output_paths[0]).parent.mkdir(parents=True, exist_ok=True)

def load_processed():
    df = pd.read_csv(processed_input)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return X, y


MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "knn": KNeighborsClassifier(),
    "svc": SVC(probability=True, random_state=42),
    "naive_bayes": GaussianNB(),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "perceptron": Perceptron(random_state=42),
}

def train_and_save(model, output_path, X, y):
    model.fit(X, y)
    joblib.dump(model, output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    X, y = load_processed()
    for model_name, model in MODELS.items():
        path = Path(f"models/{model_name}.pkl")
        if path in [Path(p) for p in output_paths]:
            train_and_save(model, path, X, y)