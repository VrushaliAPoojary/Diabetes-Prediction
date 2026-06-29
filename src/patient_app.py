"""Patient-friendly Streamlit UI for real-time diabetes risk prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from train_diabetes_model import build_models, load_data, preprocess_features

DATA_PATH = Path("data/diabetes.csv")
MODEL_PATH = Path("models/diabetes_model.joblib")
DIET_DATA_PATH = Path("data/diet_recommendations.csv")
RANDOM_STATE = 42
LOW_RISK_LIMIT = 0.35
MEDIUM_RISK_LIMIT = 0.65

DIET_PREFERENCES = ["Vegetarian", "Non-vegetarian", "Eggetarian"]

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
        .risk-medium {
            color: #d97706;
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


@st.cache_data(show_spinner="Loading diet recommendation knowledge base...")
def get_diet_knowledge() -> pd.DataFrame:
    """Load the local diet recommendation dataset used by the UI recommendation engine."""
    return pd.read_csv(DIET_DATA_PATH)


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


def render_sidebar_inputs() -> tuple[dict[str, float], dict[str, object]]:
    """Render patient health and diet-personalization inputs in the sidebar."""
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
    st.sidebar.header("🥗 Diet personalization")
    diet_preference = st.sidebar.radio(
        "Food preference",
        DIET_PREFERENCES,
        help="Used only to personalize meal ideas. Vegetarian users receive vegetarian-only suggestions.",
    )
    is_pregnant = st.sidebar.checkbox(
        "Currently pregnant",
        value=False,
        help="Pregnancy needs individualized care, so the app adds safer, pregnancy-focused diet reminders.",
    )
    activity_level = st.sidebar.selectbox(
        "Daily activity level",
        ["Low", "Moderate", "High"],
        index=1,
        help="Used to tune general lifestyle suggestions.",
    )

    st.sidebar.divider()
    st.sidebar.info(
        "This app is for education and early awareness only. Please consult a qualified clinician for medical advice."
    )
    profile = {
        "diet_preference": diet_preference,
        "is_pregnant": is_pregnant,
        "activity_level": activity_level,
    }
    return patient, profile


def risk_label(probability: float) -> tuple[str, str, str]:
    """Convert prediction probability into low, medium, or high risk text and CSS class."""
    if probability < LOW_RISK_LIMIT:
        return "Low diabetes chance", "risk-low", "Keep your healthy routine consistent."
    if probability < MEDIUM_RISK_LIMIT:
        return "Medium diabetes chance", "risk-medium", "Improve food choices and monitor key health numbers."
    return "High diabetes chance", "risk-high", "Please arrange a medical checkup and review your glucose results."


def age_group(age: float) -> str:
    """Return a simple age group for diet guidance."""
    if age < 30:
        return "young adult"
    if age < 60:
        return "adult"
    return "senior adult"


def personalized_diet_plan(patient: dict[str, float], profile: dict[str, object], risk_level: str) -> dict[str, list[str]]:
    """Create an AI-style data-driven diet plan from risk tier, age, pregnancy, and food preference."""
    preference = str(profile["diet_preference"])
    pregnant = "Yes" if bool(profile["is_pregnant"]) else "No"
    activity = str(profile["activity_level"])
    group = age_group(patient["Age"])
    risk_key = risk_level.split()[0]

    knowledge = get_diet_knowledge()
    matches = knowledge[
        knowledge["diet_preference"].isin(["Any", preference])
        & knowledge["age_group"].isin(["Any", group])
        & knowledge["pregnancy"].isin(["Any", pregnant])
        & knowledge["risk_level"].isin(["Any", risk_key])
        & knowledge["activity_level"].isin(["Any", activity])
    ]

    plan: dict[str, list[str]] = {}
    for category, rows in matches.groupby("category", sort=False):
        plan[category] = rows["recommendation"].drop_duplicates().tolist()
    return plan


def render_diet_plan(patient: dict[str, float], profile: dict[str, object], risk_level: str) -> None:
    """Render personalized diet suggestions and doctor-consultation guidance."""
    st.subheader("🥗 AI-guided personalized diet plan")
    st.caption(
        "This is a data-driven educational recommendation engine using a local diet knowledge base, your risk tier, "
        "age, pregnancy status, activity level, and vegetarian/non-vegetarian preference."
    )
    st.write(
        f"Plan type: **{profile['diet_preference']}** • Age group: **{age_group(patient['Age']).title()}** • "
        f"Pregnancy: **{'Yes' if profile['is_pregnant'] else 'No'}** • Risk tier: **{risk_level}**"
    )

    plan = personalized_diet_plan(patient, profile, risk_level)
    columns = st.columns(2)
    for index, (section, items) in enumerate(plan.items()):
        with columns[index % 2]:
            with st.expander(section, expanded=True):
                for item in items:
                    st.markdown(f"- {item}")

    st.warning(
        "Important: This diet plan is educational and not a medical prescription. Diabetes, pregnancy, kidney disease, "
        "heart disease, food allergies, medicines, and insulin use can change what is safe for you. Please consult a "
        "doctor, registered dietitian, or diabetes educator before following any diet or treatment plan."
    )


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


def render_prediction(patient: dict[str, float], profile: dict[str, object]) -> None:
    """Render prediction results, comparison data, diet guidance, and next steps."""
    data = get_dataset()
    model = get_model()
    patient_frame = pd.DataFrame([patient])
    probability = float(model.predict_proba(patient_frame)[0][1])
    prediction = int(probability >= 0.5)
    label, css_class, risk_message = risk_label(probability)

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
        st.caption(risk_message)
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
    render_diet_plan(patient, profile, label)

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
patient_inputs, patient_profile = render_sidebar_inputs()
render_prediction(patient_inputs, patient_profile)
