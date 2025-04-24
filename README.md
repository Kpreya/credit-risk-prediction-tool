# Credit Risk Assessment Project

An intelligent credit risk assessment system that uses machine learning to predict credit risks and provide explainable AI insights.

## Features

- Credit risk prediction using Random Forest model
- Interactive web interface built with Streamlit
- Real-time risk assessment and probability scores
- AI explanations using SHAP values
- Risk mitigation suggestions
- Support for various loan purposes and applicant profiles

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kpreya/Credit-Risk-Prediction-tool.git
cd Credit-Risk-Prediction-tool
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Credit-Risk-Project/
├── data/                   # Data files
│   └── german_credit_data.csv
├── models/                 # Trained models
│   ├── preprocessor.joblib
│   ├── random_forest.joblib
│   └── label_encoder.joblib
├── notebooks/             # Jupyter notebooks
│   └── creditriskanalysis.ipynb
├── plots/                 # Generated plots
├── app/                   # Application files
├── config.py             # Configuration settings
├── credit_risk_app.py    # Streamlit application
└── requirements.txt      # Project dependencies
```

## Usage

1. Run the Jupyter notebook to train models:
```bash
jupyter notebook notebooks/creditriskanalysis.ipynb
```

2. Start the Streamlit application:
```bash
streamlit run credit_risk_app.py
```

3. Access the web interface at `http://localhost:8501`

## Model Features

The system considers various factors including:
- Age
- Gender
- Job Level
- Housing Status
- Savings Account
- Checking Account
- Loan Amount
- Loan Duration
- Loan Purpose

## Technical Details

- Python 3.8+
- Main libraries: scikit-learn, TensorFlow, Streamlit, SHAP
- Models: Random Forest, Gradient Boosting, CNN
- Data preprocessing: Standard scaling, one-hot encoding
- Class imbalance handling: SMOTE

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
