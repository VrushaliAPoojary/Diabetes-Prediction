"""Patient-friendly Streamlit UI for real-time diabetes risk prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_diabetes_model import build_models, load_data, preprocess_features

DATA_PATH = Path("data/diabetes.csv")
MODEL_PATH = Path("models/diabetes_model.joblib")
RANDOM_STATE = 42

FEATURE_HELP = {
    "Pregnancies": "Number of pregnancies. Use 0 if this does not apply.",
    "Glucose": "Plasma glucose concentration. This is usually one of the strongest diabetes indicators.",
    "BloodPressure": "Diastolic blood pressure in mm Hg.",
    "SkinThickness": "Triceps skin-fold thickness in mm.",
    "Insulin": "2-hour serum insulin in mu U/ml.",
    "BMI": "Body Mass Index calculated from height and weight.",
    "DiabetesPedigreeFunction": "Family-history score for diabetes likelihood.",
    "Age": "Patient age in years.",
}

DEFAULT_VALUES = {
    "Pregnancies": 1,
    "Glucose": 120,
    "BloodPressure": 72,
    "SkinThickness": 23,
    "Insulin": 80,
    "BMI": 28.0,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 33,
}

INPUT_LIMITS = {
    "Pregnancies": (0, 17, 1),
    "Glucose": (40, 220, 1),
    "BloodPressure": (30, 130, 1),
    "SkinThickness": (5, 80, 1),
    "Insulin": (10, 850, 1),
    "BMI": (12.0, 70.0, 0.1),
    "DiabetesPedigreeFunction": (0.05, 2.50, 0.01),
    "Age": (18, 90, 1),
}


def configure_page() -> None:
    """Configure Streamlit page settings and custom styling."""
    st.set_page_config(
        page_title="Diabetes Risk Predictor",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(135deg, #f8fbff 0%, #eef7f5 100%);
        }
        .hero-card {
            padding: 2rem;
            border-radius: 26px;
            background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
            color: white;
            box-shadow: 0 20px 45px rgba(37, 99, 235, 0.18);
            margin-bottom: 1.5rem;
        }
        .hero-card h1 {
            font-size: 2.65rem;
            margin-bottom: 0.4rem;
        }
        .hero-card p {
            font-size: 1.08rem;
            opacity: 0.95;
        }
        .metric-card {
            padding: 1.25rem;
            border-radius: 22px;
            background: white;
            border: 1px solid #dbeafe;
            box-shadow: 0 12px 30px rgba(15, 118, 110, 0.08);
        }
        .risk-low {
            color: #047857;
            font-weight: 800;
        }
        .risk-high {
            color: #dc2626;
            font-weight: 800;
        }
        .gentle-note {
            padding: 1rem;
            border-radius: 18px;
            background: #ecfeff;
            border: 1px solid #a5f3fc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading diabetes reference dataset...")
def get_dataset() -> pd.DataFrame:
    """Load the diabetes dataset for training and patient comparison."""
    return load_data(DATA_PATH)


@st.cache_resource(show_spinner="Preparing prediction model...")
def get_model() -> object:
    """Load a saved model or train a Random Forest model when no saved model exists."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    data = get_dataset()
    x, y = preprocess_features(data)
    model = build_models(RANDOM_STATE)["Random Forest"]
    model.fit(x, y)
    return model


def render_sidebar_inputs() -> dict[str, float]:
    """Render patient input controls in the sidebar."""
    st.sidebar.header("🧾 Patient health inputs")
    st.sidebar.caption("Move the sliders or type values. Prediction updates instantly.")

    patient: dict[str, float] = {}
    for feature, default in DEFAULT_VALUES.items():
        minimum, maximum, step = INPUT_LIMITS[feature]
        patient[feature] = st.sidebar.number_input(
            feature,
            min_value=minimum,
            max_value=maximum,
            value=default,
            step=step,
            help=FEATURE_HELP[feature],
        )

    st.sidebar.divider()
    st.sidebar.info(
        "This app is for education and early awareness only. Please consult a qualified clinician for medical advice."
    )
    return patient


def risk_label(probability: float) -> tuple[str, str]:
    """Convert prediction probability into a patient-friendly label and CSS class."""
    if probability >= 0.5:
        return "Higher diabetes risk", "risk-high"
    return "Lower diabetes risk", "risk-low"


def comparison_table(data: pd.DataFrame, patient: dict[str, float]) -> pd.DataFrame:
    """Compare the patient's values with dataset medians and percentiles."""
    rows = []
    for feature, value in patient.items():
        percentile = float((data[feature] <= value).mean() * 100)
        rows.append(
            {
                "Health measure": feature,
                "Your value": round(float(value), 3),
                "Dataset median": round(float(data[feature].median()), 3),
                "Your percentile": f"{percentile:.0f}%",
            }
        )
    return pd.DataFrame(rows)


def render_prediction(patient: dict[str, float]) -> None:
    """Render prediction results, comparison data, and next-step guidance."""
    data = get_dataset()
    model = get_model()
    patient_frame = pd.DataFrame([patient])
    probability = float(model.predict_proba(patient_frame)[0][1])
    prediction = int(probability >= 0.5)
    label, css_class = risk_label(probability)

    st.markdown(
        """
        <div class="hero-card">
            <h1>🩺 Diabetes Risk Predictor</h1>
            <p>Enter patient health values and get an instant, easy-to-understand diabetes risk estimate.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_col, gauge_col, note_col = st.columns([1.15, 1, 1])
    with result_col:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Prediction result")
        st.markdown(f'<h2 class="{css_class}">{label}</h2>', unsafe_allow_html=True)
        st.write("Model output:", "Diabetes likely" if prediction else "Diabetes not likely")
        st.markdown("</div>", unsafe_allow_html=True)

    with gauge_col:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Estimated risk score")
        st.metric("Probability", f"{probability * 100:.1f}%")
        st.progress(min(max(probability, 0.0), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)

    with note_col:
        st.markdown('<div class="gentle-note">', unsafe_allow_html=True)
        st.subheader("What to do next")
        st.write(
            "Use this as a learning tool. If your risk appears high, consider speaking with a healthcare professional "
            "and reviewing glucose, BMI, diet, physical activity, and family history."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("📊 Your values compared with the reference dataset")
    st.dataframe(comparison_table(data, patient), use_container_width=True, hide_index=True)

    st.subheader("How this prediction is made")
    st.write(
        "The app uses the same preprocessing and machine-learning pipeline as the training project. "
        "It compares your entered values with patterns learned from the diabetes dataset, then estimates the "
        "probability of the diabetes class."
    )


configure_page()
render_prediction(render_sidebar_inputs())
