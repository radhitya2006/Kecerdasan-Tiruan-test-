"""Run the YieldPredict demo with the trained pipeline."""

from pathlib import Path
from pickle import UnpicklingError

import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from train import FEATURE_COLUMNS, MODEL_PATH


@st.cache_resource(show_spinner=False)
def load_pipeline(model_path: Path, model_revision: int) -> Pipeline:
    """Load the pipeline again when its file revision changes."""
    return joblib.load(model_path)


def main() -> None:
    st.set_page_config(page_title="YieldPredict", layout="centered")
    st.title("YieldPredict")
    st.write("Perkirakan hasil panen per hektare dari data negara dan kondisi pertanian.")

    try:
        pipeline = load_pipeline(MODEL_PATH, MODEL_PATH.stat().st_mtime_ns)
    except (OSError, EOFError, UnpicklingError, ValueError, KeyError):
        st.error("File model belum tersedia atau gagal dibaca. Buat ulang dari folder proyek.")
        st.code("& '.venv/Scripts/python.exe' train.py", language="powershell")
        return

    encoder = pipeline.named_steps["preprocessing"].named_transformers_["kategori"]
    countries, crops = (list(categories) for categories in encoder.categories_)
    has_example = "Indonesia" in countries and "Rice, paddy" in crops

    with st.form("prediksi_panen"):
        if has_example:
            st.caption("Nilai awal memakai contoh padi Indonesia tahun 2013.")

        country_column, crop_column = st.columns(2)
        area = country_column.selectbox(
            "Negara", countries, index=countries.index("Indonesia") if has_example else 0
        )
        item = crop_column.selectbox(
            "Tanaman", crops, index=crops.index("Rice, paddy") if has_example else 0
        )
        year = st.number_input(
            "Tahun", min_value=1, value=2013, step=1, help="Model diuji pada data 2010–2013."
        )

        rain_column, pesticide_column = st.columns(2)
        rainfall = rain_column.number_input(
            "Curah hujan tahunan (mm)", min_value=0.0, value=2702.0 if has_example else 0.0, step=1.0
        )
        pesticides = pesticide_column.number_input(
            "Penggunaan pestisida (ton)",
            min_value=0.0,
            value=1597.0 if has_example else 0.0,
            step=1.0,
            help="Masukkan total penggunaan pestisida negara tersebut dalam setahun.",
        )
        submitted = st.form_submit_button("Prediksi hasil panen", type="primary", width="stretch")

    if not submitted:
        return

    features = pd.DataFrame([[area, item, year, rainfall, pesticides]], columns=FEATURE_COLUMNS)
    try:
        yield_ton_ha = pipeline.predict(features).item() / 10_000
    except ValueError:
        st.error("Prediksi gagal. Periksa kembali nilai yang dimasukkan.")
        return

    result = f"{yield_ton_ha:.3f}".replace(".", ",")
    st.metric("Prediksi hasil panen", f"{result} ton/ha", border=True)


if __name__ == "__main__":
    main()
