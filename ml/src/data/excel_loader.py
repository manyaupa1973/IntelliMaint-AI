
 
##    """Excel ingestion placeholder."""
    
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "machine_id",
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "rpm",
    "current_a",
    "voltage_v",
    "operating_hours",
    "failure",
    "fault_type",
]


def load_machine_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load machine sensor data from an Excel file and validate its structure.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    df = pd.read_excel(
        file_path,
        sheet_name="Sensor_Data",
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print a basic validation summary for the loaded dataset.
    """

    print()
    print("===== IntelliMaint-AI Data Summary =====")
    print()

    print(f"Rows           : {len(df)}")
    print(f"Columns        : {len(df.columns)}")
    print(f"Machines       : {df['machine_id'].nunique()}")

    print()
    print("Column names:")
    print(list(df.columns))

    print()
    print("Missing values:")
    print(df.isnull().sum())

    print()
    print("Failure distribution:")
    print(df["failure"].value_counts(dropna=False))

    print()
    print("Fault type distribution:")
    print(df["fault_type"].value_counts(dropna=False))

    print()
    print("First 5 rows:")
    print(df.head())


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[3]

    excel_file = (
        project_root
        / "data"
        / "sample"
        / "machine_sensor_data.xlsx"
    )

    machine_data = load_machine_data(excel_file)

    print_data_summary(machine_data)
   
