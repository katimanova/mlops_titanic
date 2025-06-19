import pandas as pd
import requests
import time
from sklearn.metrics import accuracy_score

# Пути к данным
TEST_DATA_PATH = "data/raw/test.csv"
TARGET_PATH = "data/raw/gender_submission.csv"
API_URL = "http://localhost:8000/predict"

# Загрузка и подготовка данных
df = pd.read_csv(TEST_DATA_PATH)
targets = pd.read_csv(TARGET_PATH).set_index("PassengerId")["Survived"]

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Преобразуем типы
df = df.astype({col: float if dt.kind == 'f' else str for col, dt in df.dtypes.items()})

# Прогнозы и метрики
predictions = []
true_labels = []
timings = []

passengers = df.to_dict(orient="records")

for i, passenger in enumerate(passengers, 1):
    passenger_id = int(passenger["PassengerId"])
    true_label = int(targets.get(passenger_id, -1))
    true_labels.append(true_label)

    start = time.time()
    response = requests.post(API_URL, json=passenger)
    elapsed = time.time() - start
    timings.append(elapsed)

    if response.status_code == 200:
        result = response.json()
        pred = result['prediction']
        predictions.append(pred)
        print(f"{i:03d}. PassengerId {passenger_id} → Prediction: {pred} | True: {true_label} | Time: {elapsed:.3f}s")
    else:
        predictions.append(-1)
        print(f"{i:03d}. ERROR: {response.status_code} - {response.text}")

# Метрики
valid_preds = [p for p, t in zip(predictions, true_labels) if t != -1]
valid_trues = [t for t in true_labels if t != -1]

accuracy = accuracy_score(valid_trues, valid_preds)
avg_latency = sum(timings) / len(timings)
print(f"Accuracy: {accuracy:.4f}")
print(f"Среднее время отклика: {avg_latency:.3f} сек")