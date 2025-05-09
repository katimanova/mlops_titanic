import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

RAW_PATH = Path("data/raw/train.csv")
PROCESSED_PATH = Path("data/processed/train_preprocessed.csv")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)


def load_processed():
    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return X, y


def load_raw():
    df = pd.read_csv(RAW_PATH)

    # простая предобработка
    df = df[["Survived", "Pclass", "Sex", "SibSp", "Parch", "Fare"]].copy()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = df.dropna()
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


def train_and_save(model, model_name, dataset_name, X, y):
    model.fit(X, y)
    path = MODEL_PATH / f"{model_name}_{dataset_name}.pkl"
    joblib.dump(model, path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    for dataset_name, loader in [("processed", load_processed), ("raw", load_raw)]:
        X, y = loader()
        for model_name, model in MODELS.items():
            train_and_save(model, model_name, dataset_name, X, y)
