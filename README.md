# Diabetes Prediction Mini Project

This project trains supervised machine-learning models to predict whether a patient is likely to have diabetes. It uses the Pima Indians Diabetes Dataset and compares Logistic Regression, Decision Tree, and Random Forest classifiers.

> Educational note: this project is for learning machine learning only. It is not a medical diagnostic tool.

## Project structure

```text
Diabetes-Prediction/
├── api/predict.py                # Vercel-compatible prediction API
├── public/index.html             # Vercel-compatible patient web UI
├── data/                         # Dataset location plus diet recommendation knowledge base
├── models/                       # Saved trained model output
├── reports/                      # Text report and confusion-matrix image output
├── src/train_diabetes_model.py   # Main training and evaluation script
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Vercel Python entrypoint configuration
├── vercel.json                   # Vercel routing configuration
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

The Streamlit app opens in your browser and lets you enter patient values in real time. It shows low, medium, or high diabetes chance, an estimated probability score, dataset comparisons, and an AI-guided diet plan based on pregnancy status, age group, activity level, and vegetarian/non-vegetarian preference.


### 6. Run the Vercel-compatible UI locally

The project keeps the Streamlit app and also includes a Vercel-ready browser UI plus Python API. To test the Vercel version locally, install the Vercel CLI and run:

```bash
npm install -g vercel
vercel dev
```

Then open the local URL printed by Vercel, usually `http://localhost:3000`. The browser UI calls `/api/predict`, which returns the diabetes chance, low/medium/high risk tier, personalized diet plan, and doctor-consultation note.

### 7. Deploy to Vercel

1. Push this repository to GitHub.
2. Open Vercel and choose **Add New Project**.
3. Import this repository.
4. Use the default settings or choose **Other** as the framework preset.
5. Deploy.

Vercel will serve `public/index.html` as the patient-friendly UI and `api/predict.py` as the Python prediction endpoint. `pyproject.toml` points Vercel to the Python function entrypoint `api.predict:handler`. The original Streamlit app remains available for local use with `streamlit run src/patient_app.py`.

### 8. Review generated outputs

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
5. **Patient-friendly UI**: Runs a Streamlit app that accepts live patient inputs and displays low, medium, or high risk predictions plus dataset comparisons.
6. **Vercel-compatible UI**: Provides `public/index.html` and `api/predict.py` so the same prediction and diet features can run as a static browser UI backed by a Python API endpoint.
7. **Diet recommendation feature**: Uses `data/diet_recommendations.csv` as a local knowledge base to suggest vegetarian, non-vegetarian, eggetarian, pregnancy-aware, age-aware, and risk-aware meal ideas.
8. **Model saving**: Saves the best-performing pipeline as `models/diabetes_model.joblib`.

## AI-guided diet recommendation feature

The web app includes an educational, data-driven diet recommendation engine. It asks for:

- Food preference: vegetarian, non-vegetarian, or eggetarian.
- Pregnancy status, so pregnant patients receive safer pregnancy-focused reminders.
- Age, which is converted into a young adult, adult, or senior adult group.
- Activity level, so lifestyle suggestions can be adjusted.
- Model probability, which is converted into low, medium, or high diabetes chance.

The diet suggestions come from `data/diet_recommendations.csv`. To improve or customize the diet data yourself:

1. Open `data/diet_recommendations.csv`.
2. Add a new row with these columns: `category`, `diet_preference`, `age_group`, `pregnancy`, `risk_level`, `activity_level`, and `recommendation`.
3. Use `Any` in a column when the recommendation should apply to everyone.
4. Use `Vegetarian`, `Non-vegetarian`, or `Eggetarian` to target a specific food preference.
5. Use `Yes` or `No` in `pregnancy` for pregnancy-specific advice.
6. Use `Low`, `Medium`, or `High` in `risk_level` for risk-specific advice.
7. Save the file and restart the Streamlit app with `streamlit run src/patient_app.py`.

Important: the diet output is educational and should not replace advice from a doctor, registered dietitian, or diabetes educator, especially for pregnancy, medication use, kidney disease, heart disease, allergies, or insulin therapy.

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
