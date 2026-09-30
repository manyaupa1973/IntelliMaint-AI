"""Data validation placeholder."""
from pathlib import Path

import pandas as pd


SENSOR_RANGES = {
    "temperature_c": (-20, 150),
    "vibration_mm_s": (0, 50),
    "pressure_bar": (0, 20),
    "rpm": (0, 10000),
    "current_a": (0, 500),
    "voltage_v": (0, 1000),
    "operating_hours": (0, 200000),
}


def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Perform structural and quality checks on machine sensor data.
    """

    report = {}

    # --------------------------------------------------
    # Basic shape
    # --------------------------------------------------

    report["rows"] = len(df)
    report["columns"] = len(df.columns)

    # --------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------

    duplicate_count = df.duplicated().sum()

    report["duplicate_rows"] = int(duplicate_count)

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    missing_values = df.isnull().sum()

    report["missing_values"] = missing_values.to_dict()

    # --------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------

    invalid_timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    ).isna().sum()

    report["invalid_timestamps"] = int(invalid_timestamps)

    # --------------------------------------------------
    # Machine ID validation
    # --------------------------------------------------

    invalid_machine_ids = (
        df["machine_id"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    report["invalid_machine_ids"] = int(invalid_machine_ids)

    # --------------------------------------------------
    # Sensor range validation
    # --------------------------------------------------

    range_violations = {}

    for column, (minimum, maximum) in SENSOR_RANGES.items():

        invalid_count = (
            (df[column] < minimum)
            | (df[column] > maximum)
        ).sum()

        range_violations[column] = int(invalid_count)

    report["range_violations"] = range_violations

    # --------------------------------------------------
    # Failure label validation
    # --------------------------------------------------

    invalid_failure_labels = (
        ~df["failure"].isin([0, 1])
    ).sum()

    report["invalid_failure_labels"] = int(
        invalid_failure_labels
    )

    # --------------------------------------------------
    # Fault label validation
    # --------------------------------------------------

    allowed_fault_types = {
        "Normal",
        "Bearing Failure",
        "Overheating",
        "Electrical Fault",
        "Pressure Fault",
    }

    invalid_fault_types = (
        ~df["fault_type"].isin(allowed_fault_types)
    ).sum()

    report["invalid_fault_types"] = int(
        invalid_fault_types
    )

    # --------------------------------------------------
    # Logical consistency check
    # --------------------------------------------------

    inconsistent_normal = (
        (df["failure"] == 1)
        & (df["fault_type"] == "Normal")
    ).sum()

    inconsistent_failure = (
        (df["failure"] == 0)
        & (df["fault_type"] != "Normal")
    ).sum()

    report["failure_fault_inconsistency"] = int(
        inconsistent_normal + inconsistent_failure
    )

    return report


def print_validation_report(report: dict) -> None:
    """
    Print the validation report in readable form.
    """

    print()
    print("===== IntelliMaint-AI Validation Report =====")
    print()

    print(f"Rows                : {report['rows']}")
    print(f"Columns             : {report['columns']}")
    print(f"Duplicate rows      : {report['duplicate_rows']}")
    print(f"Invalid timestamps  : {report['invalid_timestamps']}")
    print(f"Invalid machine IDs : {report['invalid_machine_ids']}")

    print()
    print("Missing values:")

    for column, count in report["missing_values"].items():
        print(f"{column:20} : {count}")

    print()
    print("Sensor range violations:")

    for column, count in report["range_violations"].items():
        print(f"{column:20} : {count}")

    print()
    print(
        "Invalid failure labels :",
        report["invalid_failure_labels"],
    )

    print(
        "Invalid fault types    :",
        report["invalid_fault_types"],
    )

    print(
        "Failure/fault mismatch :",
        report["failure_fault_inconsistency"],
    )


if __name__ == "__main__":

    from excel_loader import load_machine_data

    project_root = Path(__file__).resolve().parents[3]

    excel_file = (
        project_root
        / "data"
        / "sample"
        / "machine_sensor_data.xlsx"
    )

    df = load_machine_data(excel_file)

    report = validate_dataframe(df)

    print_validation_report(report)