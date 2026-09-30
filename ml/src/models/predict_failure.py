from pathlib import Path
import json

import joblib
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_FILE = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
    / "failure_prediction_model.joblib"
)

DEMO_CONFIG_FILE = (
    PROJECT_ROOT
    / "config"
    / "demo_machine.json"
)


# ---------------------------------------------------------
# Features used by the trained ML model
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


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_FILE}\n\n"
            "Please run failure_prediction.py first."
        )

    print("Loading trained ML model...")

    model = joblib.load(
        MODEL_FILE
    )

    print("Model loaded successfully.")

    return model


# ---------------------------------------------------------
# Load machine data from JSON configuration
# ---------------------------------------------------------

def load_machine_config():

    if not DEMO_CONFIG_FILE.exists():

        raise FileNotFoundError(
            f"Machine configuration file not found:\n"
            f"{DEMO_CONFIG_FILE}"
        )

    print("Loading machine input configuration...")

    with open(
        DEMO_CONFIG_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        machine_data = json.load(file)

    return machine_data


# ---------------------------------------------------------
# Validate machine input
# ---------------------------------------------------------

def validate_machine_data(
    machine_data,
):

    required_fields = [
        "machine_id",
        "temperature_c",
        "vibration_mm_s",
        "pressure_bar",
        "rpm",
        "current_a",
        "voltage_v",
        "operating_hours",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in machine_data
    ]

    if missing_fields:

        raise ValueError(
            "Missing required fields: "
            + ", ".join(missing_fields)
        )


# ---------------------------------------------------------
# Prepare ML input
# ---------------------------------------------------------

def prepare_input(
    machine_data,
):

    input_data = pd.DataFrame(
        [machine_data]
    )

    return input_data[
        FEATURE_COLUMNS
    ]


# ---------------------------------------------------------
# Make prediction
# ---------------------------------------------------------

def predict_failure(
    model,
    input_data,
):

    prediction = model.predict(
        input_data
    )[0]

    probabilities = model.predict_proba(
        input_data
    )[0]

    normal_probability = probabilities[0]

    failure_probability = probabilities[1]

    return (
        prediction,
        normal_probability,
        failure_probability,
    )


# ---------------------------------------------------------
# Display prediction result
# ---------------------------------------------------------

def display_result(
    machine_data,
    prediction,
    normal_probability,
    failure_probability,
):

    print()
    print("=" * 60)

    print(
        "              IntelliMaint-AI"
    )

    print(
        "        MACHINE FAILURE PREDICTION"
    )

    print("=" * 60)

    print()

    print(
        f"Machine ID       : "
        f"{machine_data['machine_id']}"
    )

    print(
        f"Temperature      : "
        f"{machine_data['temperature_c']} °C"
    )

    print(
        f"Vibration        : "
        f"{machine_data['vibration_mm_s']} mm/s"
    )

    print(
        f"Pressure         : "
        f"{machine_data['pressure_bar']} bar"
    )

    print(
        f"RPM              : "
        f"{machine_data['rpm']}"
    )

    print(
        f"Current          : "
        f"{machine_data['current_a']} A"
    )

    print(
        f"Voltage          : "
        f"{machine_data['voltage_v']} V"
    )

    print(
        f"Operating Hours  : "
        f"{machine_data['operating_hours']}"
    )

    print()
    print("-" * 60)
    print()

    if prediction == 1:

        prediction_text = "FAILURE"

    else:

        prediction_text = "NORMAL"

    print(
        f"Prediction          : "
        f"{prediction_text}"
    )

    print(
        f"Normal Probability  : "
        f"{normal_probability * 100:.2f}%"
    )

    print(
        f"Failure Probability : "
        f"{failure_probability * 100:.2f}%"
    )

    print()

    print("-" * 60)

    if prediction == 1:

        print(
            "ALERT: Potential machine failure detected."
        )

        print(
            "Recommendation: Investigate machine "
            "condition and consider maintenance."
        )

    else:

        print(
            "Machine condition appears normal."
        )

    print()

    print("=" * 60)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print()
    print(
        "Starting IntelliMaint-AI ML Prediction Demo..."
    )

    # Load trained model
    model = load_model()

    # Load machine input
    machine_data = load_machine_config()

    # Validate input
    validate_machine_data(
        machine_data
    )

    # Prepare input for ML model
    input_data = prepare_input(
        machine_data
    )

    # Predict
    (
        prediction,
        normal_probability,
        failure_probability,
    ) = predict_failure(
        model,
        input_data,
    )

    # Display result
    display_result(
        machine_data,
        prediction,
        normal_probability,
        failure_probability,
    )


if __name__ == "__main__":

    main()
