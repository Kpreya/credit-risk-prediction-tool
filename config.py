import os

# Project directory structure
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
APP_DIR = os.path.join(PROJECT_ROOT, 'app')
DEPLOYMENT_DIR = os.path.join(PROJECT_ROOT, 'deployment')

# Data files
GERMAN_CREDIT_DATA = os.path.join(DATA_DIR, 'german_credit_data.csv')
PREPROCESSED_DATA = os.path.join(DATA_DIR, 'preprocessed_credit_data.csv')

# Model files
LABEL_ENCODER = os.path.join(MODELS_DIR, 'label_encoder.joblib')
PREPROCESSOR = os.path.join(MODELS_DIR, 'preprocessor.joblib')
BEST_MODEL = os.path.join(MODELS_DIR, 'best_model.joblib')