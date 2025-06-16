# Snakefile

rule all:
    input:
        "results/metrics.txt"

rule preprocess:
    input:
        train="data/raw/train.csv",
        test="data/raw/test.csv"
    output:
        train_p="data/processed/train_preprocessed.csv",
        test_p="data/processed/test_preprocessed.csv"
    script:
        "src/mlops_titanic/preprocess.py"

rule train:
    input:
        train_p="data/processed/train_preprocessed.csv",
        test_p="data/processed/test_preprocessed.csv"
    output:
        models=expand("models/{model}.pkl",
               model=["logistic_regression", "knn", "svc", "naive_bayes", "decision_tree", "random_forest", "perceptron"])
    script:
        "src/mlops_titanic/train.py"

rule evaluate:
    input:
        models=expand("models/{model}.pkl",
                      model=["logistic_regression", "knn", "svc", "naive_bayes", "decision_tree", "random_forest", "perceptron"]),
        test_processed="data/processed/test_preprocessed.csv",
        target="data/raw/gender_submission.csv"
    output:
        "results/metrics.txt"
    script:
        "src/mlops_titanic/evaluate_models.py"