"""
api.py
======
Flask REST API — Web Attack Detection (self-contained edition)

Uses the same core.py module as app.py, so both stay in sync automatically.

Run:
    pip install flask pandas numpy scikit-learn joblib
    python api.py
"""

import time
import logging

from flask import Flask, request, jsonify, abort

from core import (
    CLASS_ORDER, LABEL_MAP, SEVERITY,
    generate_synthetic_dataset, train_model, predict, try_load_real_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)

    artifacts = try_load_real_model()
    bundle = None

    if artifacts:
        model = artifacts["model"]
        metadata = artifacts["metadata"]
        logger.info("Real model artifacts loaded from %s.", artifacts["source"])
    else:
        logger.warning("No real model found — training a demo model on synthetic data.")
        df_synth = generate_synthetic_dataset()
        bundle = train_model(df_synth)
        model = bundle["model"]
        metadata = bundle["metadata"]

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "model_source": "real" if artifacts else "synthetic-demo",
            "timestamp": time.time(),
        })

    @app.route("/model/info", methods=["GET"])
    def model_info():
        return jsonify(metadata)

    @app.route("/classes", methods=["GET"])
    def classes():
        return jsonify({"label_map": LABEL_MAP, "classes": CLASS_ORDER})

    @app.route("/retrain", methods=["POST"])
    def retrain():
        nonlocal model, metadata, bundle
        body = request.get_json(force=True, silent=True) or {}
        n_per_class = int(body.get("n_per_class", 250))
        label_noise = float(body.get("label_noise", 0.03))

        df_synth = generate_synthetic_dataset(n_per_class=n_per_class, label_noise=label_noise)
        bundle = train_model(df_synth)
        model = bundle["model"]
        metadata = bundle["metadata"]
        logger.info("Retrained demo model on %d synthetic examples.", len(df_synth))
        return jsonify({"status": "retrained", "metadata": metadata})

    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        body = request.get_json(force=True, silent=True)
        if not body or "payloads" not in body:
            abort(400, "Request body must contain 'payloads' list.")

        payloads = body["payloads"]
        if not isinstance(payloads, list) or not payloads:
            abort(400, "'payloads' must be a non-empty list of strings.")
        if len(payloads) > 500:
            abort(400, "Maximum 500 payloads per request.")

        t0 = time.perf_counter()
        result = predict(payloads, model)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        predictions = []
        for i, pl in enumerate(payloads):
            lbl = result["labels"][i]
            sev_name, _, sev_score = SEVERITY.get(lbl, ("UNKNOWN", "#94a3b8", 50))
            item = {
                "payload": pl,
                "label": lbl,
                "severity": sev_name,
                "severity_score": sev_score,
            }
            if result["confidence"]:
                item["confidence"] = result["confidence"][i]
            if result["probabilities"]:
                item["probabilities"] = result["probabilities"][i]
            predictions.append(item)

        logger.info("Predicted %d payload(s) in %.2fms", len(payloads), latency_ms)

        return jsonify({
            "predictions": predictions,
            "model": metadata.get("best_model", "unknown"),
            "latency_ms": latency_ms,
            "count": len(predictions),
        })

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": str(e)}), 400

    @app.errorhandler(503)
    def service_unavailable(e):
        return jsonify({"error": str(e)}), 503

    return app


if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Web Attack Detection API on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
