# classification_project
Standardized classification project with data preprocessing, Shap feature selection, Catboost model, Optuna HO, MLFlow logging, SHAP explanations.
Pre-commit configured with black, flake8 and mypy hooks was used for code checking and formatting.

Dataset used: sklearn breast cancer dataset

## Installation
Install dependencies from requirements.txt. Adjust parameters in config.yml (initial parameters for Optuna HO and training)

## Project structure
```
classification_project/
|   config.yml
|   README.md
|   requirements.txt
|   
└─── output/
    └─── figures/
    └─── models/
|   
└─── data/
|
└─── src/
    └─── mlruns/
    |    utils.py
    |    project_dirs.py

```

- folder `output`: figures, models, predictions
- folder `data`: preprocessed dataset
- folder `src`: code, mlflow runs

## config.yml parameters

# Use









