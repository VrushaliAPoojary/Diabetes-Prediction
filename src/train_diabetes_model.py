"""Train and evaluate diabetes prediction models on the Pima Indians dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def download_dataset(path: Path) -> None:
    """Download the Pima Indians Diabetes Dataset if it is not already present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {path} ...")
    urlretrieve(DATA_URL, path)


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset from disk, downloading it first when needed."""
    if not path.exists():
        download_dataset(path)
    data = pd.read_csv(path)
    expected = {"Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"}
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return data


def preprocess_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split features/target and mark medically impossible zero values as missing."""
    clean = data.copy()
    clean[ZERO_AS_MISSING] = clean[ZERO_AS_MISSING].replace(0, np.nan)
    x = clean.drop(columns="Outcome")
    y = clean["Outcome"]
    return x, y


def build_models(random_state: int) -> dict[str, Pipeline]:
    """Create candidate supervised-learning pipelines."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DecisionTreeClassifier(max_depth=5, random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=random_state)),
            ]
        ),
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, model_name: str, output_path: Path) -> None:
    """Save a confusion-matrix image for the selected model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["No Diabetes", "Diabetes"])
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_and_evaluate(args: argparse.Namespace) -> None:
    """Train candidate models, print metrics, and save the best model/report files."""
    data = load_data(args.data_path)
    x, y = preprocess_features(data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    models = build_models(args.random_state)
    results: list[tuple[str, float, Pipeline, np.ndarray, str]] = []

    for name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, target_names=["No Diabetes", "Diabetes"])
        results.append((name, accuracy, pipeline, predictions, report))
        print(f"\n{name}\n{'-' * len(name)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(report)

    best_name, best_accuracy, best_model, best_predictions, best_report = max(results, key=lambda item: item[1])

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.model_path)
    save_confusion_matrix(y_test, best_predictions, best_name, args.confusion_matrix_path)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", encoding="utf-8") as report_file:
        report_file.write(f"Best model: {best_name}\n")
        report_file.write(f"Accuracy: {best_accuracy:.4f}\n\n")
        report_file.write("Classification report:\n")
        report_file.write(best_report)

    print(f"\nBest model: {best_name} ({best_accuracy:.4f})")
    print(f"Saved trained model to {args.model_path}")
    print(f"Saved text report to {args.report_path}")
    print(f"Saved confusion matrix to {args.confusion_matrix_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diabetes prediction models.")
    parser.add_argument("--data-path", type=Path, default=Path("data/diabetes.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("models/diabetes_model.joblib"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/model_report.txt"))
    parser.add_argument("--confusion-matrix-path", type=Path, default=Path("reports/confusion_matrix.png"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    train_and_evaluate(parse_args())
