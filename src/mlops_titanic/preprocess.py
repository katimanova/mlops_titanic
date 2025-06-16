import pandas as pd
from pathlib import Path

# Доступ к input/output из Snakefile
train_input = snakemake.input.train      # "data/raw/train.csv"
test_input = snakemake.input.test        # "data/raw/test.csv"
train_output = snakemake.output.train_p  # "data/processed/train_preprocessed.csv"
test_output = snakemake.output.test_p    # "data/processed/test_preprocessed.csv"

# Убедимся, что выходная директория существует
Path(train_output).parent.mkdir(parents=True, exist_ok=True)

def preprocess(df: pd.DataFrame, is_train: bool=True) -> pd.DataFrame:
    df = df.copy()

    # Кодировка пола
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Заполнение пропущенных значений
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Embarked"] = df["Embarked"].astype(str).map({"S": 0, "C": 1, "Q": 2}).astype(int)

    # Новая фича
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Удалим ненужные колонки
    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Удалим NaN из таргета, если train
    if is_train:
        df = df.dropna(subset=["Survived"])

    return df

if __name__ == "__main__":
    train_df = pd.read_csv(train_input)
    test_df = pd.read_csv(test_input)

    train_processed = preprocess(train_df, is_train=True)
    test_processed = preprocess(test_df, is_train=False)

    train_processed.to_csv(train_output, index=False)
    test_processed.to_csv(test_output, index=False)