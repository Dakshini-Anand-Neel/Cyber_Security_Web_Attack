"""
tests/test_predict.py
=====================
Unit tests for the unified inference layer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import unittest.mock as mock

import numpy as np
import pytest

from src.inference import (
    ATTACK_LIBRARY,
    CLASS_ORDER,
    extract_features,
    predict,
    shannon_entropy,
)


class TestFeatureExtraction:
    PAYLOADS = [
        "<script>alert(1)</script>",
        "1' UNION SELECT null,null--",
        "What color is the sky?",
    ]

    def test_shape(self):
        X = extract_features(self.PAYLOADS)
        assert X.shape == (3, 7)

    def test_all_finite(self):
        X = extract_features(self.PAYLOADS)
        assert np.isfinite(X.values).all()

    def test_xss_keyword_flag(self):
        X = extract_features(["<script>alert(1)</script>", "hello"])
        assert X.iloc[0]["kw_xss"] == 1
        assert X.iloc[1]["kw_xss"] == 0

    def test_sql_keyword_flag(self):
        X = extract_features(["1 UNION SELECT null--", "hello"])
        assert X.iloc[0]["kw_sql"] == 1
        assert X.iloc[1]["kw_sql"] == 0

    def test_payload_len(self):
        p = "Hello World"
        X = extract_features([p])
        assert X.iloc[0]["payload_len"] == len(p)

    def test_entropy_positive(self):
        assert shannon_entropy("abcdef") > 0

    def test_empty_payload(self):
        X = extract_features([""])
        assert X.iloc[0]["payload_entropy"] == 0.0


class TestPredict:
    def test_predict_with_mock(self):
        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.05, 0.03, 0.92]])

        mock_scaler = mock.MagicMock()
        mock_scaler.transform.side_effect = lambda x: x

        mock_encoder = mock.MagicMock()
        mock_encoder.inverse_transform.return_value = np.array([3])
        mock_encoder.classes_ = np.array([1, 2, 3])

        artifacts = {
            "model": mock_model,
            "scaler": mock_scaler,
            "target_encoder": mock_encoder,
            "metadata": {},
        }
        result = predict(["hello world"], artifacts)
        assert result["labels"] == ["Normal"]
        assert result["confidence"] == [92.0]


class TestAttackLibrary:
    def test_all_classes_present(self):
        for cls in CLASS_ORDER:
            assert cls in ATTACK_LIBRARY
            assert len(ATTACK_LIBRARY[cls]) >= 3


class TestFlaskAPI:
    @pytest.fixture
    def client(self):
        from app.api import create_app

        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 2])
        mock_model.predict_proba.return_value = np.array([
            [0.95, 0.03, 0.02],
            [0.01, 0.97, 0.02],
            [0.02, 0.01, 0.97],
        ])

        mock_scaler = mock.MagicMock()
        mock_scaler.transform.side_effect = lambda x: x

        mock_encoder = mock.MagicMock()
        mock_encoder.inverse_transform.return_value = np.array([1, 2, 3])
        mock_encoder.classes_ = np.array([1, 2, 3])

        mock_meta = {"best_model": "LogisticRegression", "final_accuracy": 1.0, "final_f1": 1.0}

        with mock.patch("app.api.load_artifacts", return_value={
            "model": mock_model,
            "scaler": mock_scaler,
            "target_encoder": mock_encoder,
            "metadata": mock_meta,
        }):
            app = create_app()
            app.config["TESTING"] = True
            yield app.test_client()

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_classes_endpoint(self, client):
        r = client.get("/classes")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "classes" in data

    def test_predict_with_probabilities(self, client):
        r = client.post("/predict", json={"payloads": ["<script>alert(1)</script>"]})
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "predictions" in data
        assert "probabilities" in data["predictions"][0]
        assert "severity" in data["predictions"][0]

    def test_predict_missing_body(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 400
