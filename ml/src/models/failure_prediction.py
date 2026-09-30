"""Failure prediction placeholder."""
from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "machine_sensor_data.xlsx"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
    / "failure_prediction_model.joblib"
)


# ---------------------------------------------------------
# Import our Excel loader
# ---------------------------------------------------------

DATA_MODULE_PATH = (
    PROJECT_ROOT
    / "ml"
    / "src"
    / "data"
)

sys.path.insert(
    0,
    str(DATA_MODULE_PATH),
)

from excel_loader import load_machine_data


# ---------------------------------------------------------
# Features and target
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "rpm",
    "current_a",
    "voltage_v",
    "operating_hours",
]

TARGET_COLUMN = "failure"


# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------

def prepare_data(df: pd.DataFrame):

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
) -> None:

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print()
    print("===== MODEL PERFORMANCE =====")
    print()

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print()
    print("===== CONFUSION MATRIX =====")
    print()

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print(matrix)

    print()
    print("===== CLASSIFICATION REPORT =====")
    print()

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Normal",
                "Failure",
            ],
        )
    )


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

def print_feature_importance(
    model: RandomForestClassifier,
) -> None:

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(
        ascending=False
    )

    print()
    print("===== FEATURE IMPORTANCE =====")
    print()

    for feature, score in importance.items():

        print(
            f"{feature:20} : "
            f"{score:.4f}"
        )


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

def save_model(
    model: RandomForestClassifier,
) -> None:

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print()
    print(
        f"Model saved to: {MODEL_FILE}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print()
    print(
        "Loading IntelliMaint-AI machine data..."
    )

    df = load_machine_data(
        DATA_FILE
    )

    print(
        f"Dataset loaded: {len(df)} records"
    )

    print()
    print(
        "Preparing training and testing data..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_data(df)

    print(
        f"Training records : {len(X_train)}"
    )

    print(
        f"Testing records  : {len(X_test)}"
    )

    print()
    print(
        "Training Random Forest model..."
    )

    model = train_model(
        X_train,
        y_train,
    )

    print(
        "Model training completed."
    )

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    print_feature_importance(
        model
    )

    save_model(
        model
    )

    print()
    print(
        "Failure prediction training completed successfully."
    )


if __name__ == "__main__":
    main()