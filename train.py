"""Train and save the YieldPredict pipeline selected in the notebook."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "yield_df.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "yield_pipeline.joblib"

CATEGORICAL_COLUMNS = ["Area", "Item"]
HARVEST_KEY = [*CATEGORICAL_COLUMNS, "Year"]
FEATURE_COLUMNS = [
    *HARVEST_KEY,
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
]
TARGET_COLUMN = "hg/ha_yield"
TRAIN_YEARS = (1990, 2006)
VALIDATION_YEARS = (2007, 2009)


def load_training_data(data_path: Path) -> pd.DataFrame:
    columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    data = pd.read_csv(data_path, usecols=columns)[columns]
    data = data.drop_duplicates(ignore_index=True)

    if data.isna().any().any():
        raise ValueError("The model features and target must contain no missing values.")
    if data.duplicated(subset=HARVEST_KEY).any():
        raise ValueError("Conflicting values for the same country, crop, and year.")

    # Match the notebook's row order when refitting on train and validation.
    years = data["Year"]
    train = data.loc[years.between(*TRAIN_YEARS)]
    validation = data.loc[years.between(*VALIDATION_YEARS)]
    if train.empty or validation.empty:
        raise ValueError("The train and validation year ranges must both contain records.")
    return pd.concat([train, validation])


def build_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        [("kategori", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)],
        remainder="passthrough",
    )
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])


def save_pipeline(pipeline: Pipeline, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=model_path.parent, suffix=".joblib", delete=False) as file:
        temporary_path = Path(file.name)

    try:
        joblib.dump(pipeline, temporary_path, compress=3)
        temporary_path.replace(model_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    training = load_training_data(DATA_PATH)
    pipeline = build_pipeline()
    pipeline.fit(training[FEATURE_COLUMNS], training[TARGET_COLUMN])
    save_pipeline(pipeline, MODEL_PATH)

    print(f"Trained on {len(training):,} records from {TRAIN_YEARS[0]} to {VALIDATION_YEARS[1]}.")
    print(f"Saved pipeline to {MODEL_PATH.relative_to(PROJECT_ROOT)}.")


if __name__ == "__main__":
    main()
