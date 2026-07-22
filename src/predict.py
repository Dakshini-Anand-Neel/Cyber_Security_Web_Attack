"""
src/predict.py
==============
CLI inference helper — load a saved model and predict on new payloads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.inference import load_artifacts, predict


def predict_payloads(
    payloads: List[str],
    artifacts: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Predict attack type for a list of payload strings."""
    if artifacts is None:
        artifacts = load_artifacts()
        if artifacts is None:
            raise FileNotFoundError(
                "Model artifacts not found. Run notebooks/01_eda.ipynb first."
            )

    result = predict(payloads, artifacts)
    out = pd.DataFrame({"Payload": payloads, "predicted_label": result["labels"]})
    if result["confidence"]:
        out["confidence"] = result["confidence"]
    return out


if __name__ == "__main__":
    samples = [
        "Select a paint color for the room.",
        "1' OR '1'='1'; DROP TABLE users;--",
        "<script>alert('XSS')</script>",
    ]
    print(predict_payloads(samples).to_string(index=False))
