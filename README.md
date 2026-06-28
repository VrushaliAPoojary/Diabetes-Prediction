# Diabetes Prediction Mini Project

This project trains supervised machine-learning models to predict whether a patient is likely to have diabetes. It uses the Pima Indians Diabetes Dataset and compares Logistic Regression, Decision Tree, and Random Forest classifiers.

> Educational note: this project is for learning machine learning only. It is not a medical diagnostic tool.

## Project structure

```text
Diabetes-Prediction/
├── data/                         # Dataset location; script downloads diabetes.csv here
├── models/                       # Saved trained model output
├── reports/                      # Text report and confusion-matrix image output
├── src/train_diabetes_model.py   # Main training and evaluation script
├── requirements.txt              # Python dependencies
└── README.md                     # Instructions and report
```

## Dataset

The script uses the Pima Indians Diabetes Dataset, a binary classification dataset with these columns:

| Column | Meaning |
| --- | --- |
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin-fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Family-history diabetes score |
| Age | Patient age |
| Outcome | Target label: `0` = no diabetes, `1` = diabetes |

Some medical measurements cannot realistically be zero, so the training script treats zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` as missing values and imputes them with the median.

## Step-by-step commands

Run these commands from the project root.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train and evaluate the models

```bash
python src/train_diabetes_model.py
```

The script will download `data/diabetes.csv` automatically if the file is missing, train all candidate models, print accuracy and classification reports, and save the best model.


### 5. Run the patient-friendly web UI

```bash
streamlit run src/patient_app.py
```

The Streamlit app opens in your browser and lets you enter patient values in real time. It shows a diabetes risk prediction, an estimated probability score, and a comparison of the entered values against the dataset medians and percentiles.

### 6. Review generated outputs

After training or using the app, check these files:

```text
models/diabetes_model.joblib
reports/model_report.txt
reports/confusion_matrix.png
```

## What the code does

1. **Data loading**: Loads `data/diabetes.csv`; downloads it automatically when missing.
2. **Data preprocessing**: Replaces impossible zero medical measurements with missing values, imputes medians, and scales features for Logistic Regression.
3. **Model training**: Trains Logistic Regression, Decision Tree, and Random Forest classifiers.
4. **Model evaluation**: Calculates accuracy, confusion matrix, precision, recall, F1-score, and support.
5. **Patient-friendly UI**: Runs a Streamlit app that accepts live patient inputs and displays risk predictions plus dataset comparisons.
6. **Model saving**: Saves the best-performing pipeline as `models/diabetes_model.joblib`.

## Report template

When you submit or explain the mini project, use this short report format.

### Aim

To build a machine-learning model that predicts diabetes status from patient health measurements.

### Algorithm used

The project compares Logistic Regression, Decision Tree, and Random Forest models. The best model is selected using test-set accuracy.

### Evaluation metrics

- Accuracy: overall proportion of correct predictions.
- Confusion matrix: counts true positives, true negatives, false positives, and false negatives.
- Classification report: precision, recall, F1-score, and support for both classes.

### Expected output

The terminal displays model metrics, and the project creates a saved model, a text report, and a confusion-matrix image inside the `models/` and `reports/` folders.

## Optional custom commands

Use a different train/test split:

```bash
python src/train_diabetes_model.py --test-size 0.25
```

Save outputs to custom paths:

```bash
python src/train_diabetes_model.py --model-path models/my_model.joblib --report-path reports/my_report.txt
```
