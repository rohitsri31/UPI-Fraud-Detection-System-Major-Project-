"""Trains a small model once, then checks the saved files and the HTTP API."""
import tempfile
import unittest
from pathlib import Path

from app import create_app
from src import generate_data, train

HIGH_RISK = {"amount": 24000, "typical_amount": 600, "hour": 3, "txn_type": "p2p",
             "is_new_payee": True, "is_new_device": True, "is_new_city": True, "txns_last_hour": 3}
LOW_RISK = {"amount": 450, "typical_amount": 500, "hour": 14, "txn_type": "merchant"}


class PipelineAndApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.model_path = cls.tmp / "model.joblib"
        df = generate_data.generate(n_users=400, n_transactions=12000, fraud_rate=0.03, seed=7)
        cls.summary = train.train_and_save(df, models=["logreg", "random_forest"],
                                           strategies=["class_weight", "smote"],
                                           model_path=cls.model_path, report_dir=cls.tmp / "reports",
                                           verbose=False)
        cls.client = create_app(cls.model_path, cls.tmp / "app.db").test_client()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_training_writes_all_outputs(self):
        for name in ("model.joblib", "comparison.csv", "metrics.json"):
            self.assertTrue((self.tmp / name).exists(), name)
        for name in ("roc_curve.png", "confusion_matrix.png", "model_comparison.png"):
            self.assertTrue((self.tmp / "reports" / name).exists(), name)

    def test_model_beats_the_rule_baseline(self):
        self.assertGreater(self.summary["test"]["recall"], self.summary["rule_baseline"]["recall"])
        self.assertGreater(self.summary["test"]["roc_auc"], 0.9)

    def test_health(self):
        body = self.client.get("/api/health").get_json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])

    def test_risky_payment_scores_higher_and_is_flagged(self):
        high = self.client.post("/api/score", json=HIGH_RISK).get_json()
        low = self.client.post("/api/score", json=LOW_RISK).get_json()
        self.assertGreater(high["fraud_probability"], low["fraud_probability"])
        self.assertTrue(high["flagged"])
        self.assertFalse(low["flagged"])
        self.assertIn("Unrecognised device", high["indicators"])
        self.assertEqual(low["indicators"], [])

    def test_bad_requests_get_400(self):
        for payload in ({}, {"amount": 500, "hour": 99}, {"amount": 500, "txn_type": "cash"}):
            with self.subTest(payload=payload):
                response = self.client.post("/api/score", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())
        response = self.client.post("/api/score", data="not json", content_type="text/plain")
        self.assertEqual(response.status_code, 400)

    def test_scored_transactions_are_stored_and_listed(self):
        self.client.post("/api/score", json=HIGH_RISK)
        self.client.post("/api/score", json=LOW_RISK)
        stats = self.client.get("/api/stats").get_json()
        self.assertGreaterEqual(stats["total"], 2)
        self.assertGreaterEqual(stats["flagged"], 1)
        flagged = self.client.get("/api/transactions?flagged=1").get_json()
        self.assertTrue(flagged)
        self.assertTrue(all(item["flagged"] for item in flagged))
        self.assertEqual(self.client.get("/api/transactions?limit=abc").status_code, 400)

    def test_dashboard_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"UPI Fraud Monitor", response.data)

    def test_missing_model_gives_503_not_a_crash(self):
        client = create_app(self.tmp / "missing.joblib", self.tmp / "other.db").test_client()
        self.assertFalse(client.get("/api/health").get_json()["model_loaded"])
        self.assertEqual(client.post("/api/score", json=LOW_RISK).status_code, 503)


if __name__ == "__main__":
    unittest.main()
