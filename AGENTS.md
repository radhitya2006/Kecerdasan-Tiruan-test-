# Repository Guidelines

## Project Structure

- `train.py` prepares the data, trains the selected pipeline, and saves it to `models/yield_pipeline.joblib`. The notebook contains the model evaluation. `app.py` loads the pipeline for the Streamlit demo.
- `data/raw/yield_df.csv` is the original Kaggle dataset. Place derived datasets in `data/processed/` and exploratory notebooks in `notebooks/`.
- Save fitted pipelines in `models/`. Git ignores `models/*.joblib`.

## Local Development

Run these commands from the repository root with Python 3.11.

- `py -3.11 -m venv .venv` creates an isolated environment on Windows. Select `.venv/Scripts/python.exe` as your editor's Python interpreter.
- `& '.venv/Scripts/python.exe' -m pip install -r requirements.txt` installs the project dependencies from PowerShell.
- `python -m py_compile app.py train.py` checks both entry points for syntax errors.
- `& '.venv/Scripts/python.exe' train.py` trains on the 1990–2009 records and saves the pipeline.
- `& '.venv/Scripts/python.exe' -m streamlit run app.py` starts the local demo. Run training first to generate the model file.

## Coding Style and Data Handling

Use four spaces for indentation, `snake_case` for functions and variables, and `PascalCase` for classes. The repository has no configured formatter or linter. Keep exploratory code in notebooks and put reusable preprocessing in the training pipeline so Streamlit can use the same transformations.

Preserve the original CSV. Exclude its unnamed row index and the target `hg/ha_yield` from the model inputs. For future-year evaluation, hold out later years and fit preprocessing on the training split only. Document changes to duplicate handling, feature units, and evaluation splits.

## Testing

The repository has no test suite or coverage threshold yet. For the first automated tests, use `unittest` and name files `tests/test_*.py`. Run them with `python -m unittest discover -s tests` once the directory exists. Use small fixtures to check preprocessing and prediction inputs.

## Commits and Pull Requests

History uses short subjects such as `Resolve README merge conflict`. Use a concise imperative subject. In pull requests, summarize the change, list verification commands, and link any related issue. Include screenshots for visible Streamlit changes and report the evaluation split and metric for model changes.
