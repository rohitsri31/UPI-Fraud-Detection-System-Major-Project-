import unittest

import pandas as pd

from src import config, features, generate_data, imbalance


def make_txns(receivers, minutes, amounts=None, sender="U1"):
    t0 = pd.Timestamp("2026-04-01 10:00")
    n = len(receivers)
    return pd.DataFrame({
        "timestamp": [t0 + pd.Timedelta(minutes=m) for m in minutes],
        "sender_id": [sender] * n,
        "receiver_id": receivers,
        "amount": amounts or [100.0] * n,
        "txn_type": ["p2p"] * n,
        "device_id": ["D1"] * n,
        "city": ["X"] * n,
        "is_fraud": [0] * n,
    })


class GeneratorTests(unittest.TestCase):
    def test_shape_rate_and_order(self):
        df = generate_data.generate(n_users=300, n_transactions=6000, fraud_rate=0.03, seed=1)
        self.assertEqual(len(df), 6000)
        self.assertEqual(int(df["is_fraud"].sum()), 180)
        for column in config.RAW_COLUMNS:
            self.assertIn(column, df.columns)
        self.assertTrue(df["timestamp"].is_monotonic_increasing)
        self.assertTrue(set(df["txn_type"]) <= set(config.TXN_TYPES))

    def test_same_seed_gives_same_data(self):
        a = generate_data.generate(n_users=100, n_transactions=1000, seed=5)
        b = generate_data.generate(n_users=100, n_transactions=1000, seed=5)
        pd.testing.assert_frame_equal(a, b)


class HistoryFeatureTests(unittest.TestCase):
    def test_velocity_and_new_payee(self):
        history = features.add_history_features(make_txns(["A", "A", "B", "B"], [0, 10, 20, 180]))
        self.assertEqual(list(history["txns_last_hour"]), [0, 1, 2, 0])
        self.assertEqual(list(history["is_new_payee"]), [0, 0, 1, 0])

    def test_first_payment_has_no_history_flags(self):
        history = features.add_history_features(make_txns(["A"], [0]))
        flags = history.loc[0, ["is_new_payee", "is_new_device", "is_new_city"]]
        self.assertEqual(int(flags.sum()), 0)

    def test_new_device_after_history(self):
        df = make_txns(["A", "A"], [0, 30])
        df.loc[1, "device_id"] = "D2"
        history = features.add_history_features(df)
        self.assertEqual(list(history["is_new_device"]), [0, 1])

    def test_typical_amount_uses_only_the_past(self):
        history = features.add_history_features(make_txns(["A", "A"], [0, 30], amounts=[100.0, 1000.0]))
        median = 550.0   # median of the two amounts; used until a sender has history
        self.assertAlmostEqual(history.loc[0, "typical_amount"], median)
        self.assertAlmostEqual(history.loc[1, "typical_amount"], (100.0 + 3 * median) / 4)

    def test_feature_matrix_has_expected_columns(self):
        df = generate_data.generate(n_users=200, n_transactions=3000, seed=2)
        X, y = features.build_features(df)
        self.assertEqual(list(X.columns), config.FEATURE_COLUMNS)
        self.assertFalse(X.isna().any().any())
        self.assertEqual(len(X), len(y))


class RequestFeatureTests(unittest.TestCase):
    def test_defaults(self):
        X = features.features_from_request({"amount": 500})
        self.assertEqual(list(X.columns), config.FEATURE_COLUMNS)
        row = X.iloc[0]
        self.assertEqual(row["amount_ratio"], 1.0)
        self.assertEqual(row["hour"], 12)
        self.assertEqual(row["type_p2p"], 1.0)
        self.assertEqual(row["is_new_payee"], 0.0)

    def test_flags_accept_booleans_and_strings(self):
        X = features.features_from_request({"amount": 500, "is_new_payee": True, "is_new_device": "true",
                                            "is_new_city": "no", "hour": "3"})
        row = X.iloc[0]
        self.assertEqual((row["is_new_payee"], row["is_new_device"], row["is_new_city"]), (1.0, 1.0, 0.0))
        self.assertEqual(row["is_night"], 1.0)

    def test_bad_input_is_rejected(self):
        bad = [{}, {"amount": "abc"}, {"amount": 0}, {"amount": 500, "hour": 30},
               {"amount": 500, "txn_type": "cash"}, {"amount": 500, "txns_last_hour": -1}]
        for payload in bad:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    features.features_from_request(payload)


class ImbalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = generate_data.generate(n_users=300, n_transactions=8000, fraud_rate=0.02, seed=3)
        cls.X, cls.y = features.build_features(df)

    def test_smote_adds_valid_fraud_rows(self):
        X2, y2 = imbalance.smote(self.X, self.y)
        self.assertGreater(len(X2), len(self.X))
        self.assertAlmostEqual(y2.mean(), 0.2 / 1.2, places=2)
        pd.testing.assert_frame_equal(X2.iloc[:len(self.X)], self.X.reset_index(drop=True))
        for column in ("is_new_payee", "is_night", "type_p2p"):
            self.assertTrue(set(X2[column].unique()) <= {0.0, 1.0})
        self.assertTrue(((X2["hour"] % 1) == 0).all())

    def test_undersample_keeps_all_fraud(self):
        X2, y2 = imbalance.undersample(self.X, self.y)
        self.assertEqual(int(y2.sum()), int(self.y.sum()))
        self.assertAlmostEqual(y2.mean(), 0.1 / 1.1, places=2)


if __name__ == "__main__":
    unittest.main()
