"""Vercel-compatible diabetes prediction API endpoint."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "diabetes.csv"
DIET_DATA_PATH = ROOT / "data" / "diet_recommendations.csv"
MODEL_PATH = ROOT / "models" / "diabetes_model.joblib"
INDEX_PATH = ROOT / "public" / "index.html"
RANDOM_STATE = 42
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
LOW_RISK_LIMIT = 0.35
MEDIUM_RISK_LIMIT = 0.65
EMBEDDED_INDEX_HTML = '<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="UTF-8" />\n  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n  <title>Diabetes Risk Predictor</title>\n  <style>\n    :root {\n      --teal: #0f766e;\n      --blue: #2563eb;\n      --green: #047857;\n      --amber: #d97706;\n      --red: #dc2626;\n      --ink: #0f172a;\n      --muted: #64748b;\n      --card: rgba(255, 255, 255, 0.92);\n    }\n\n    * { box-sizing: border-box; }\n\n    body {\n      margin: 0;\n      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;\n      color: var(--ink);\n      background:\n        radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 32rem),\n        linear-gradient(135deg, #f8fbff 0%, #eef7f5 100%);\n      min-height: 100vh;\n    }\n\n    .page { max-width: 1180px; margin: 0 auto; padding: 32px 18px 52px; }\n\n    .hero {\n      border-radius: 30px;\n      padding: 34px;\n      color: white;\n      background: linear-gradient(135deg, var(--teal), var(--blue));\n      box-shadow: 0 24px 60px rgba(37, 99, 235, 0.22);\n      margin-bottom: 24px;\n    }\n\n    .hero h1 { font-size: clamp(2rem, 5vw, 4rem); margin: 0 0 10px; }\n    .hero p { max-width: 780px; margin: 0; font-size: 1.08rem; opacity: 0.96; line-height: 1.7; }\n\n    .grid { display: grid; grid-template-columns: minmax(320px, 0.9fr) minmax(320px, 1.1fr); gap: 24px; align-items: start; }\n    @media (max-width: 880px) { .grid { grid-template-columns: 1fr; } }\n\n    .card {\n      background: var(--card);\n      border: 1px solid rgba(148, 163, 184, 0.24);\n      border-radius: 24px;\n      padding: 24px;\n      box-shadow: 0 18px 45px rgba(15, 118, 110, 0.08);\n      backdrop-filter: blur(16px);\n    }\n\n    .card h2 { margin-top: 0; }\n    .form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }\n    @media (max-width: 560px) { .form-grid { grid-template-columns: 1fr; } }\n\n    label { display: block; font-weight: 700; margin-bottom: 6px; }\n    small { color: var(--muted); line-height: 1.4; }\n\n    input, select {\n      width: 100%;\n      border: 1px solid #cbd5e1;\n      border-radius: 14px;\n      padding: 12px 13px;\n      font-size: 0.98rem;\n      background: white;\n      outline-color: var(--teal);\n    }\n\n    .field { display: grid; gap: 4px; }\n\n    button {\n      width: 100%;\n      margin-top: 22px;\n      border: none;\n      border-radius: 16px;\n      padding: 15px 18px;\n      color: white;\n      font-weight: 800;\n      font-size: 1.02rem;\n      cursor: pointer;\n      background: linear-gradient(135deg, var(--teal), var(--blue));\n      box-shadow: 0 14px 32px rgba(37, 99, 235, 0.20);\n    }\n\n    button:disabled { opacity: 0.7; cursor: wait; }\n\n    .result-card { display: none; }\n    .risk-pill { display: inline-flex; align-items: center; border-radius: 999px; padding: 9px 14px; font-weight: 900; margin-bottom: 10px; }\n    .risk-low { color: var(--green); background: #dcfce7; }\n    .risk-medium { color: var(--amber); background: #fef3c7; }\n    .risk-high { color: var(--red); background: #fee2e2; }\n\n    .probability { font-size: 3rem; font-weight: 900; margin: 8px 0 0; }\n    .bar { height: 14px; border-radius: 999px; background: #e2e8f0; overflow: hidden; margin: 14px 0; }\n    .bar span { display: block; height: 100%; width: 0%; background: linear-gradient(90deg, var(--green), var(--amber), var(--red)); transition: width 0.4s ease; }\n\n    .diet-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 18px; }\n    @media (max-width: 640px) { .diet-grid { grid-template-columns: 1fr; } }\n    .diet-section { border-radius: 18px; background: #f8fafc; border: 1px solid #e2e8f0; padding: 16px; }\n    .diet-section h3 { margin-top: 0; color: var(--teal); }\n    .diet-section li { margin-bottom: 8px; line-height: 1.5; }\n\n    .note { border-radius: 18px; padding: 16px; background: #ecfeff; border: 1px solid #a5f3fc; line-height: 1.6; }\n    .warning { border-radius: 18px; padding: 16px; background: #fff7ed; border: 1px solid #fed7aa; line-height: 1.6; margin-top: 18px; }\n    .error { color: var(--red); background: #fee2e2; border: 1px solid #fecaca; padding: 14px; border-radius: 16px; display: none; margin-top: 16px; }\n  </style>\n</head>\n<body>\n  <main class="page">\n    <section class="hero">\n      <h1>🩺 Diabetes Risk Predictor</h1>\n      <p>Enter patient health values to get a real-time diabetes chance estimate, low/medium/high risk tier, and an educational personalized diet plan based on age, pregnancy status, activity level, and food preference.</p>\n    </section>\n\n    <section class="grid">\n      <form class="card" id="predictionForm">\n        <h2>Patient inputs</h2>\n        <div class="form-grid">\n          <div class="field"><label>Pregnancies</label><input name="Pregnancies" type="number" min="0" max="17" step="1" value="1" required><small>Use 0 if not applicable.</small></div>\n          <div class="field"><label>Glucose</label><input name="Glucose" type="number" min="40" max="220" step="1" value="120" required><small>Plasma glucose concentration.</small></div>\n          <div class="field"><label>Blood Pressure</label><input name="BloodPressure" type="number" min="30" max="130" step="1" value="72" required><small>Diastolic blood pressure.</small></div>\n          <div class="field"><label>Skin Thickness</label><input name="SkinThickness" type="number" min="5" max="80" step="1" value="23" required><small>Triceps skin-fold thickness.</small></div>\n          <div class="field"><label>Insulin</label><input name="Insulin" type="number" min="10" max="850" step="1" value="80" required><small>2-hour serum insulin.</small></div>\n          <div class="field"><label>BMI</label><input name="BMI" type="number" min="12" max="70" step="0.1" value="28" required><small>Body Mass Index.</small></div>\n          <div class="field"><label>Diabetes Pedigree Function</label><input name="DiabetesPedigreeFunction" type="number" min="0.05" max="2.5" step="0.01" value="0.45" required><small>Family-history score.</small></div>\n          <div class="field"><label>Age</label><input name="Age" type="number" min="18" max="90" step="1" value="33" required><small>Patient age in years.</small></div>\n          <div class="field"><label>Food preference</label><select name="diet_preference"><option>Vegetarian</option><option>Non-vegetarian</option><option>Eggetarian</option></select><small>Vegetarian users receive vegetarian-only suggestions.</small></div>\n          <div class="field"><label>Currently pregnant?</label><select name="is_pregnant"><option value="false">No</option><option value="true">Yes</option></select><small>Adds pregnancy-focused reminders.</small></div>\n          <div class="field"><label>Activity level</label><select name="activity_level"><option>Low</option><option selected>Moderate</option><option>High</option></select><small>Used for lifestyle suggestions.</small></div>\n        </div>\n        <button id="submitButton" type="submit">Predict diabetes chance</button>\n        <div class="error" id="errorBox"></div>\n      </form>\n\n      <section class="card">\n        <h2>Prediction result</h2>\n        <div class="note" id="emptyState">Fill the form and click the button to see the prediction, probability score, and personalized diet plan.</div>\n        <div class="result-card" id="resultCard">\n          <div id="riskPill" class="risk-pill"></div>\n          <p id="riskMessage"></p>\n          <div class="probability" id="probabilityText"></div>\n          <div class="bar"><span id="probabilityBar"></span></div>\n          <p id="profileSummary"></p>\n          <h2>🥗 AI-guided personalized diet plan</h2>\n          <div class="diet-grid" id="dietGrid"></div>\n          <div class="warning" id="doctorNote"></div>\n        </div>\n      </section>\n    </section>\n  </main>\n\n  <script>\n    const form = document.getElementById("predictionForm");\n    const submitButton = document.getElementById("submitButton");\n    const errorBox = document.getElementById("errorBox");\n    const emptyState = document.getElementById("emptyState");\n    const resultCard = document.getElementById("resultCard");\n    const riskPill = document.getElementById("riskPill");\n    const riskMessage = document.getElementById("riskMessage");\n    const probabilityText = document.getElementById("probabilityText");\n    const probabilityBar = document.getElementById("probabilityBar");\n    const profileSummary = document.getElementById("profileSummary");\n    const dietGrid = document.getElementById("dietGrid");\n    const doctorNote = document.getElementById("doctorNote");\n\n    function formPayload() {\n      const formData = new FormData(form);\n      const payload = {};\n      for (const [key, value] of formData.entries()) {\n        if (["diet_preference", "activity_level"].includes(key)) {\n          payload[key] = value;\n        } else if (key === "is_pregnant") {\n          payload[key] = value === "true";\n        } else {\n          payload[key] = Number(value);\n        }\n      }\n      return payload;\n    }\n\n    function renderDietPlan(plan) {\n      dietGrid.innerHTML = "";\n      Object.entries(plan).forEach(([section, items]) => {\n        const article = document.createElement("article");\n        article.className = "diet-section";\n        const list = items.map((item) => `<li>${item}</li>`).join("");\n        article.innerHTML = `<h3>${section}</h3><ul>${list}</ul>`;\n        dietGrid.appendChild(article);\n      });\n    }\n\n    form.addEventListener("submit", async (event) => {\n      event.preventDefault();\n      submitButton.disabled = true;\n      submitButton.textContent = "Calculating...";\n      errorBox.style.display = "none";\n\n      try {\n        const payload = formPayload();\n        const response = await fetch("/api/predict", {\n          method: "POST",\n          headers: { "Content-Type": "application/json" },\n          body: JSON.stringify(payload),\n        });\n        const data = await response.json();\n        if (!response.ok) throw new Error(data.error || "Prediction failed");\n\n        const percentage = data.probability * 100;\n        emptyState.style.display = "none";\n        resultCard.style.display = "block";\n        riskPill.className = `risk-pill risk-${data.risk_class}`;\n        riskPill.textContent = data.risk;\n        riskMessage.textContent = data.risk_message;\n        probabilityText.textContent = `${percentage.toFixed(1)}%`;\n        probabilityBar.style.width = `${Math.min(Math.max(percentage, 0), 100)}%`;\n        profileSummary.innerHTML = `Plan type: <strong>${payload.diet_preference}</strong> • Age group: <strong>${data.age_group}</strong> • Pregnancy: <strong>${payload.is_pregnant ? "Yes" : "No"}</strong>`;\n        renderDietPlan(data.diet_plan);\n        doctorNote.textContent = data.doctor_note;\n      } catch (error) {\n        errorBox.textContent = error.message;\n        errorBox.style.display = "block";\n      } finally {\n        submitButton.disabled = false;\n        submitButton.textContent = "Predict diabetes chance";\n      }\n    });\n  </script>\n</body>\n</html>\n'

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def download_dataset(path: Path) -> None:
    """Download the diabetes dataset for Vercel fallback training when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(DATA_URL, path)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the diabetes dataset without importing plotting or Streamlit-only dependencies."""
    import pandas as pd

    if not path.exists():
        download_dataset(path)
    return pd.read_csv(path)


def preprocess_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features for the lightweight Vercel fallback model."""
    import numpy as np

    clean = data.copy()
    clean[ZERO_AS_MISSING] = clean[ZERO_AS_MISSING].replace(0, np.nan)
    return clean.drop(columns="Outcome"), clean["Outcome"]


def build_vercel_model() -> Pipeline:
    """Build the lightweight Random Forest pipeline used by the Vercel function."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)),
        ]
    )


def get_model() -> Any:
    """Load the trained model or train a Random Forest fallback when the model file is absent."""
    import joblib

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    data = load_dataset(DATA_PATH)
    x, y = preprocess_features(data)
    model = build_vercel_model()
    model.fit(x, y)
    return model


def risk_label(probability: float) -> tuple[str, str, str]:
    """Return a low, medium, or high patient-friendly chance label."""
    if probability < LOW_RISK_LIMIT:
        return "Low diabetes chance", "low", "Keep your healthy routine consistent."
    if probability < MEDIUM_RISK_LIMIT:
        return "Medium diabetes chance", "medium", "Improve food choices and monitor key health numbers."
    return "High diabetes chance", "high", "Please arrange a medical checkup and review your glucose results."


def age_group(age: float) -> str:
    """Return the age group used for diet guidance."""
    if age < 30:
        return "young adult"
    if age < 60:
        return "adult"
    return "senior adult"


def get_diet_plan(payload: dict[str, Any], risk: str) -> dict[str, list[str]]:
    """Build a diet plan from the local diet knowledge base."""
    import pandas as pd

    diet_data = pd.read_csv(DIET_DATA_PATH)
    preference = str(payload["diet_preference"])
    pregnancy = "Yes" if bool(payload["is_pregnant"]) else "No"
    activity = str(payload["activity_level"])
    group = age_group(float(payload["Age"]))
    risk_key = risk.split()[0]

    matches = diet_data[
        diet_data["diet_preference"].isin(["Any", preference])
        & diet_data["age_group"].isin(["Any", group])
        & diet_data["pregnancy"].isin(["Any", pregnancy])
        & diet_data["risk_level"].isin(["Any", risk_key])
        & diet_data["activity_level"].isin(["Any", activity])
    ]

    plan: dict[str, list[str]] = {}
    for category, rows in matches.groupby("category", sort=False):
        plan[category] = rows["recommendation"].drop_duplicates().tolist()
    return plan


def parse_payload(raw_body: bytes) -> dict[str, Any]:
    """Parse and validate the API request body."""
    payload = json.loads(raw_body.decode("utf-8"))
    missing = [field for field in FEATURES if field not in payload]
    missing.extend(field for field in ["diet_preference", "is_pregnant", "activity_level"] if field not in payload)
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    for field in FEATURES:
        payload[field] = float(payload[field])
    payload["diet_preference"] = str(payload["diet_preference"])
    payload["is_pregnant"] = bool(payload["is_pregnant"])
    payload["activity_level"] = str(payload["activity_level"])
    return payload


class handler(BaseHTTPRequestHandler):
    """Vercel Python Function handler."""

    def _send_json(self, status_code: int, response: dict[str, Any]) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def _send_html(self, status_code: int, html: str) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else EMBEDDED_INDEX_HTML
            self._send_html(200, html)
            return
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self._send_json(404, {"error": "Not found"})

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            payload = parse_payload(raw_body)
            import pandas as pd

            patient = {feature: payload[feature] for feature in FEATURES}
            patient_frame = pd.DataFrame([patient])
            model = get_model()
            probability = float(model.predict_proba(patient_frame)[0][1])
            risk, risk_class, risk_message = risk_label(probability)
            response = {
                "probability": probability,
                "risk": risk,
                "risk_class": risk_class,
                "risk_message": risk_message,
                "age_group": age_group(payload["Age"]),
                "diet_plan": get_diet_plan(payload, risk),
                "doctor_note": (
                    "This prediction and diet plan are educational only. Please consult a doctor, registered "
                    "dietitian, or diabetes educator, especially during pregnancy, medication use, insulin use, "
                    "kidney disease, heart disease, allergies, or any existing medical condition."
                ),
            }
            self._send_json(200, response)
        except Exception as error:
            self._send_json(400, {"error": str(error)})
