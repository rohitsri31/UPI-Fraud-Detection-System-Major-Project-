# UPI Fraud Detection: starter project

A small, runnable base for the UPI fraud detection project: synthetic data, feature engineering,
model comparison, a saved model, and a Flask API with a dashboard. It is written to be read and changed.

## Quick start

```bash
cd upi-fraud-detection
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.generate_data            # creates data/transactions.csv (60,000 synthetic rows)
python -m src.train                    # compares models, saves models/model.joblib, charts in reports/
python run.py                          # dashboard at http://127.0.0.1:8000
```

Training takes under a minute. Optional: `python -m unittest discover -s tests` runs the tests.
If the dashboard says "No model", run `python -m src.train` and restart `python run.py`.

## What is where

| Synopsis module | Files |
| --- | --- |
| Data and preprocessing | `src/generate_data.py`, `src/data.py` |
| Feature engineering | `src/features.py` |
| Model training and evaluation | `src/train.py`, `src/imbalance.py` |
| Prediction API | `app/routes.py`, `src/predict.py` |
| Dashboard and alerts | `app/templates/index.html`, `app/static/app.js`, `app/db.py` |

```
src/        data, features, training, scoring        config.py holds every setting
app/        Flask app: routes, SQLite storage, dashboard page and script
tests/      unit tests and an end-to-end API test
data/       transactions.csv and app.db appear here
models/     model.joblib, comparison.csv, metrics.json appear here
reports/    ROC curve, confusion matrix, model comparison and feature charts for your report
run.py      starts the dashboard
```

## How it works

1. **Data.** Each row is one payment: timestamp, sender, receiver, amount, type, device, city, and an `is_fraud` label.
2. **Features.** Twelve inputs per payment, such as amount compared with the user's usual amount, hour of day,
   new payee, new device, new city, and payments in the last hour. History features use only earlier payments,
   so nothing leaks from the future.
3. **Training.** The data is split by time: oldest 60% trains, next 20% validates, newest 20% tests.
   Each model is tried with each imbalance strategy (none, class weights, SMOTE, undersampling).
   The best pair is chosen by F2 on the validation data, which weighs recall above precision,
   and its decision threshold is tuned there too. The test set is scored once, next to a simple rule-based baseline.
4. **Scoring.** The API takes the same signals for one payment and returns a fraud probability, a risk level
   (low, medium, high), a flagged yes/no, and plain-language reasons. Every scored payment is saved to SQLite
   and listed on the dashboard.

## API

```bash
curl -X POST http://127.0.0.1:8000/api/score -H "Content-Type: application/json" \
  -d '{"amount": 24000, "typical_amount": 600, "hour": 3, "txn_type": "p2p",
       "is_new_payee": true, "is_new_device": true, "is_new_city": true, "txns_last_hour": 3}'
```

| Route | Purpose |
| --- | --- |
| `POST /api/score` | Score one payment. Only `amount` is required. |
| `GET /api/transactions?flagged=1&limit=25` | Recently scored payments. |
| `GET /api/stats` | Totals and flag rate. |
| `GET /api/health` | Server status and which model is loaded. |

`/api/score` fields: `amount` (INR), `typical_amount` (the user's usual payment, defaults to `amount`),
`hour` (0 to 23, default 12), `txn_type` (`p2p`, `merchant`, `collect_request`, `qr_scan`),
`is_new_payee`, `is_new_device`, `is_new_city` (true/false), `txns_last_hour`.
Bad input returns 400 with a message.

## Make it yours

| To do this | Change this |
| --- | --- |
| Use a real dataset | See "Using your own dataset" below |
| Add or remove a feature | `finalize_features` in `src/features.py`, `FEATURE_COLUMNS` in `src/config.py`, then the form in `index.html` |
| Add a model | `make_model` in `src/train.py` and `MODELS` in `src/config.py` |
| Favour precision over recall | `F_BETA` in `src/config.py` (1 is balanced, 2 favours recall) |
| Change the split or data size | `TRAIN_FRACTION`, `VAL_FRACTION`, `N_TRANSACTIONS`, `FRAUD_RATE` in `src/config.py` |
| Move to MySQL or MongoDB | The four functions in `app/db.py`; keep their names and signatures |
| Explain predictions with SHAP | Replace `risk_indicators` in `src/predict.py` |
| Change the look | `app/templates/index.html` and `app/static/style.css` |

## Using your own dataset

Put a CSV at `data/transactions.csv` with these columns, then run `python -m src.train`:

`timestamp, sender_id, receiver_id, amount, txn_type, device_id, city, is_fraud`

`txn_type` should be one of `p2p`, `merchant`, `collect_request`, `qr_scan`; other values simply get no type flag.
`is_fraud` is 0 or 1. Public datasets rarely have this exact layout, so map their columns in `src/data.py`.
If a dataset has no device or city, fill those columns with one constant value; the matching features then stay at 0.

## Things to know

- **The data is synthetic.** The scores show how well the models learn the patterns in the generator, not how they
  would do on real UPI fraud. Retrain on real or public data before quoting any accuracy.
- **Live scoring needs history signals from the caller.** `typical_amount`, `is_new_payee` and the like are sent in the
  request. In a real system they would be looked up from the user's past transactions.
- **The reasons on the dashboard are simple rules**, not the model's own attribution.
- **XGBoost is optional and has not been run yet.** Install it with `pip install -r requirements-optional.txt`
  (macOS also needs `brew install libomp`). Without it the project skips that model. If it errors on your
  machine, remove `"xgboost"` from `MODELS` in `src/config.py`.
- **SMOTE is written out in `src/imbalance.py`**, so no extra package is needed.
- Retrain after upgrading scikit-learn, and only load model files you trained yourself.
- The dashboard loads Bootstrap from a CDN, so it needs internet for styling. Flask's built-in server is for local use only.
- Port 8000 is used because macOS reserves 5000. Change it with `PORT=8001 python run.py`.
