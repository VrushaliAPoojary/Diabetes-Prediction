"""Vercel-compatible diabetes prediction API endpoint."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_diabetes_model import build_models, load_data, preprocess_features  # noqa: E402

DATA_PATH = ROOT / "data" / "diabetes.csv"
DIET_DATA_PATH = ROOT / "data" / "diet_recommendations.csv"
MODEL_PATH = ROOT / "models" / "diabetes_model.joblib"
RANDOM_STATE = 42
LOW_RISK_LIMIT = 0.35
MEDIUM_RISK_LIMIT = 0.65
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


def get_model() -> Any:
    """Load the trained model or train a Random Forest fallback when the model file is absent."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    data = load_data(DATA_PATH)
    x, y = preprocess_features(data)
    model = build_models(RANDOM_STATE)["Random Forest"]
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
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            payload = parse_payload(raw_body)
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
