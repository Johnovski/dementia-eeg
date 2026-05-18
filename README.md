# Dementia EEG Analysis

This repository contains code for EEG-based dementia analysis, including preprocessing support, feature engineering, model training, ensemble prediction, visualization, and explainability workflows.

The datasets are not included in this repository. They should be downloaded separately from OpenNeuro and kept outside Git tracking.

## Datasets

Primary dataset:

- OpenNeuro `ds004504`, version `1.0.8`
- URL: https://openneuro.org/datasets/ds004504/versions/1.0.8/metadata

Complementary test dataset:

- OpenNeuro `ds006036`, version `1.0.6`
- URL: https://openneuro.org/datasets/ds006036/versions/1.0.6

After downloading, place the datasets in local data folders such as:

```text
photic_data/
Olfactory/
```

These folders are ignored by Git through `.gitignore` so that large dataset files are not uploaded to GitHub.

## Project Structure

```text
quick_demo.py                 Quick end-to-end demonstration script
dataset_analysis.py           Dataset inspection and summary utilities
feature_engineering.py        Feature extraction and preparation
train_model.py                Model training workflow
ensemble_predict.py           Ensemble prediction workflow
explain_model.py              Model explainability utilities
advanced_shap_analysis.py     SHAP-based interpretation analysis
shap_mmse_comparison.py       SHAP and MMSE comparison analysis
plot_results.py               Result plotting utilities
visualize_features.py         Feature visualization utilities
requirements.txt              Python dependencies
```

Generated folders such as `features/`, `models/`, `results/`, `figures/`, and `preprocessed/` are also excluded from Git.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

Run the quick demo:

```powershell
python quick_demo.py
```

Train models:

```powershell
python train_model.py
```

Generate explanations:

```powershell
python explain_model.py
```

Depending on where the datasets are downloaded, update the dataset paths inside the scripts before running the full pipeline.

## Repository Notes

- This repository is intended to store source code and lightweight documentation only.
- Raw datasets, preprocessed files, trained models, and generated results should remain local.
- Dataset access, citation, and reuse should follow the terms listed on the linked OpenNeuro dataset pages.
