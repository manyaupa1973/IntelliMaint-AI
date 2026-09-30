from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


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

FIGURES_DIR = (
    PROJECT_ROOT
    / "ml"
    / "reports"
    / "figures"
)

# Allow importing our data loader
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
# Sensor columns
# ---------------------------------------------------------

SENSOR_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "rpm",
    "current_a",
    "voltage_v",
    "operating_hours",
]


# ---------------------------------------------------------
# Basic dataset information
# ---------------------------------------------------------

def print_basic_information(df: pd.DataFrame) -> None:

    print()
    print("===== DATASET OVERVIEW =====")
    print()

    print(f"Rows     : {len(df)}")
    print(f"Columns  : {len(df.columns)}")
    print(f"Machines : {df['machine_id'].nunique()}")

    print()
    print("Date range:")
    print("Start :", df["timestamp"].min())
    print("End   :", df["timestamp"].max())


# ---------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------

def print_sensor_statistics(df: pd.DataFrame) -> None:

    print()
    print("===== SENSOR STATISTICS =====")
    print()

    print(
        df[SENSOR_COLUMNS]
        .describe()
        .round(2)
    )


# ---------------------------------------------------------
# Failure distribution
# ---------------------------------------------------------

def print_failure_distribution(df: pd.DataFrame) -> None:

    print()
    print("===== FAILURE DISTRIBUTION =====")
    print()

    counts = df["failure"].value_counts().sort_index()

    percentages = (
        df["failure"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    for failure_value in counts.index:

        label = (
            "Normal"
            if failure_value == 0
            else "Failure"
        )

        print(
            f"{label:10} : "
            f"{counts[failure_value]:5} "
            f"({percentages[failure_value]:.2f}%)"
        )


# ---------------------------------------------------------
# Fault distribution
# ---------------------------------------------------------

def print_fault_distribution(df: pd.DataFrame) -> None:

    print()
    print("===== FAULT DISTRIBUTION =====")
    print()

    fault_counts = df["fault_type"].value_counts()

    print(fault_counts)


# ---------------------------------------------------------
# Compare normal vs failure sensor values
# ---------------------------------------------------------

def compare_normal_and_failure(df: pd.DataFrame) -> None:

    print()
    print("===== NORMAL VS FAILURE SENSOR MEANS =====")
    print()

    comparison = (
        df.groupby("failure")[SENSOR_COLUMNS]
        .mean()
        .round(2)
    )

    comparison.index = [
        "Normal",
        "Failure",
    ]

    print(comparison)


# ---------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------

def print_correlations(df: pd.DataFrame) -> None:

    print()
    print("===== CORRELATION WITH FAILURE =====")
    print()

    numeric_columns = SENSOR_COLUMNS + ["failure"]

    correlations = (
        df[numeric_columns]
        .corr()["failure"]
        .drop("failure")
        .sort_values(
            key=abs,
            ascending=False,
        )
    )

    print(correlations.round(3))


# ---------------------------------------------------------
# Plot failure distribution
# ---------------------------------------------------------

def plot_failure_distribution(df: pd.DataFrame) -> None:

    counts = (
        df["failure"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots()

    ax.bar(
        ["Normal", "Failure"],
        counts.values,
    )

    ax.set_title(
        "Machine Condition Distribution"
    )

    ax.set_ylabel(
        "Number of Records"
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / "failure_distribution.png"
    )

    fig.savefig(
        output_file,
        dpi=150,
    )

    plt.close(fig)

    print(
        f"Created: {output_file}"
    )


# ---------------------------------------------------------
# Plot fault distribution
# ---------------------------------------------------------

def plot_fault_distribution(df: pd.DataFrame) -> None:

    counts = df["fault_type"].value_counts()

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        counts.index,
        counts.values,
    )

    ax.set_title(
        "Fault Type Distribution"
    )

    ax.set_ylabel(
        "Number of Records"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / "fault_distribution.png"
    )

    fig.savefig(
        output_file,
        dpi=150,
    )

    plt.close(fig)

    print(
        f"Created: {output_file}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Loading IntelliMaint-AI dataset...")

    df = load_machine_data(DATA_FILE)

    print("Dataset loaded successfully.")

    print_basic_information(df)

    print_sensor_statistics(df)

    print_failure_distribution(df)

    print_fault_distribution(df)

    compare_normal_and_failure(df)

    print_correlations(df)

    print()
    print("===== GENERATING FIGURES =====")
    print()

    plot_failure_distribution(df)

    plot_fault_distribution(df)

    print()
    print("EDA completed successfully.")


if __name__ == "__main__":
    main()