"""Flask app: scoring API and dashboard."""
from flask import Flask

from src import config
from src.predict import Scorer

from . import db
from .routes import bp


def create_app(model_path=None, db_path=None):
    app = Flask(__name__)
    app.config["DB_PATH"] = str(db_path or config.DB_PATH)
    app.extensions["scorer"] = Scorer(model_path)
    db.init_db(app.config["DB_PATH"])
    app.register_blueprint(bp)
    return app
