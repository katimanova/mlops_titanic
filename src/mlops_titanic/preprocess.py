import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    return train, test


def preprocess(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    df = df.copy()

    # Кодировка пола
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Заполнение пропущенных значений
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Embarked"] = (
        df["Embarked"].astype(str).map({"S": 0, "C": 1, "Q": 2}).astype(int)
    )

    # Новая фича
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # уберем неиспользуемые признаки
    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
    df.drop(
        columns=[col for col in drop_cols if col in df.columns],
        inplace=True,
    )

    # Если train, удалим NaN из таргета
    if is_train:
        df = df.dropna(subset=["Survived"])

    return df


def main():
    train, test = load_data()

    train_processed = preprocess(train, is_train=True)
    test_processed = preprocess(test, is_train=False)

    train_processed.to_csv(PROCESSED_DIR / "train_preprocessed.csv", index=False)
    test_processed.to_csv(PROCESSED_DIR / "test_preprocessed.csv", index=False)


if __name__ == "__main__":
    main()
