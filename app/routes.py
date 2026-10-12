"""HTTP routes: the dashboard page and a small JSON API."""
from flask import Blueprint, current_app, jsonify, render_template, request

from src.predict import ModelNotLoaded

from . import db

bp = Blueprint("main", __name__)


def _scorer():
    return current_app.extensions["scorer"]


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/api/health")
def health():
    return jsonify({"status": "ok", **_scorer().info()})


@bp.post("/api/score")
def score():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error='Send a JSON object, for example {"amount": 2500, "txn_type": "p2p"}'), 400
    try:
        result = _scorer().score(payload)
    except ModelNotLoaded as err:
        return jsonify(error=str(err)), 503
    except ValueError as err:
        return jsonify(error=str(err)), 400
    result["id"] = db.save_transaction(current_app.config["DB_PATH"], float(payload["amount"]),
                                       str(payload.get("txn_type") or "p2p"), result)
    return jsonify(result)


@bp.get("/api/transactions")
def transactions():
    flagged_only = request.args.get("flagged", "").lower() in ("1", "true", "yes")
    try:
        limit = max(1, min(int(request.args.get("limit", 25)), 200))
    except ValueError:
        return jsonify(error="limit must be a whole number"), 400
    return jsonify(db.recent_transactions(current_app.config["DB_PATH"], flagged_only, limit))


@bp.get("/api/stats")
def stats():
    return jsonify(db.stats(current_app.config["DB_PATH"]))
