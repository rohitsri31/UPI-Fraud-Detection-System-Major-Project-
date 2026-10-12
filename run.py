"""Start the dashboard:  python run.py   then open http://127.0.0.1:8000"""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    # 8000, not Flask's usual 5000: macOS uses port 5000 for AirPlay Receiver.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
