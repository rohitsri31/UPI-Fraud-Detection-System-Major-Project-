"""Generate a synthetic UPI transaction dataset. No real user data is used.

Run:  python -m src.generate_data --rows 60000

Normal users pay a small circle of regular payees from their own phone and city.
Three fraud patterns are mixed in, and about 15% of fraud is deliberately "subtle"
(normal amount, own device, own city) so the problem is not trivially easy:
  takeover  account takeover: new device, other city, large amounts, often at night, bursts
  collect   fake collect requests sent to a user who then approves them
  qr        QR-code scams paid to a new merchant
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

CITIES = ["Mathura", "Agra", "Delhi", "Lucknow", "Kanpur", "Jaipur", "Mumbai", "Bengaluru"]
CITY_P = np.array([5, 3, 6, 3, 3, 3, 4, 3], dtype=float)
CITY_P = CITY_P / CITY_P.sum()

# Relative number of normal payments per hour of day (quiet at night, busy in the evening).
HOUR_W = np.array([1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 9, 10, 9, 8, 8, 9, 10, 11, 11, 10, 8, 5, 3], dtype=float)
HOUR_P = HOUR_W / HOUR_W.sum()

GENUINE_TYPE_P = {"p2p": 0.45, "merchant": 0.35, "qr_scan": 0.17, "collect_request": 0.03}


def _fmt(prefix, values, width):
    return np.char.add(prefix, np.char.zfill(np.asarray(values).astype(str), width))


class _World:
    """The made-up population: users, their usual spend, home city and regular payees."""

    def __init__(self, rng, n_users, n_days, n_payees=6000):
        self.rng, self.n_days, self.n_payees = rng, n_days, n_payees
        self.start = pd.Timestamp("2026-04-01")
        self.users = np.arange(n_users)
        self.home_city = rng.choice(CITIES, n_users, p=CITY_P)
        self.usual = np.exp(rng.normal(np.log(350), 0.7, n_users))   # typical payment size in INR
        activity = rng.gamma(2.0, 1.0, n_users)
        self.activity = activity / activity.sum()
        self.pool = [rng.choice(n_payees, int(rng.integers(4, 12)), replace=False) for _ in range(n_users)]

    def home_device(self, user):
        return f"D{user:04d}"

    def mule(self):
        """A payee outside every user's usual circle (always 'new' to the victim)."""
        return int(self.rng.integers(self.n_payees, self.n_payees + 3000))

    def random_user(self):
        return int(self.rng.choice(self.users, p=self.activity))

    def random_time(self, hour=None):
        rng = self.rng
        if hour is None:
            hour = int(rng.choice(24, p=HOUR_P))
        return self.start + pd.Timedelta(days=int(rng.integers(0, self.n_days)), hours=hour,
                                         seconds=int(rng.integers(0, 3600)))

    @staticmethod
    def row(when, user, receiver, amount, txn_type, device, city):
        return {"timestamp": when, "sender": user, "receiver": receiver, "amount": amount,
                "txn_type": txn_type, "device": device, "city": city}


def _genuine(w, n):
    rng = w.rng
    sender = rng.choice(w.users, n, p=w.activity)
    in_circle = rng.random(n) < 0.85
    circle_pick = np.fromiter((w.pool[s][rng.integers(len(w.pool[s]))] for s in sender), dtype=int, count=n)
    receiver = np.where(in_circle, circle_pick, rng.integers(0, w.n_payees, n))

    amount = w.usual[sender] * np.exp(rng.normal(0, 0.6, n))
    big = rng.random(n) < 0.02                          # occasional big genuine payment (rent, shopping)
    amount[big] *= rng.uniform(5, 12, int(big.sum()))

    base_device = _fmt("D", sender, 4)
    device = np.where(rng.random(n) < 0.05, np.char.add(base_device, "b"), base_device)   # second phone
    city = np.where(rng.random(n) < 0.06, rng.choice(CITIES, n), w.home_city[sender])      # travelling

    hours = rng.choice(24, n, p=HOUR_P)
    seconds = rng.integers(0, w.n_days, n) * 86400 + hours * 3600 + rng.integers(0, 3600, n)
    return pd.DataFrame({
        "timestamp": w.start + pd.to_timedelta(seconds, unit="s"),
        "sender_id": _fmt("U", sender, 4),
        "receiver_id": _fmt("P", receiver, 5),
        "amount": np.clip(np.round(amount, 2), 1, 100_000),
        "txn_type": rng.choice(list(GENUINE_TYPE_P), n, p=list(GENUINE_TYPE_P.values())),
        "device_id": device,
        "city": city,
    })


def _takeover_rows(w, n):
    rng, rows, made = w.rng, [], 0
    while made < n:
        user = w.random_user()
        size = min(int(rng.integers(1, 5)), n - made)          # burst of 1-4 payments
        subtle = rng.random() < 0.15
        night = (not subtle) and rng.random() < 0.6
        when = w.random_time(hour=int(rng.integers(0, 6)) if night else None)
        if subtle:
            device, city = w.home_device(user), str(w.home_city[user])
        else:
            device = f"X{int(rng.integers(0, 99999)):05d}" if rng.random() < 0.85 else w.home_device(user)
            away = [c for c in CITIES if c != w.home_city[user]]
            city = str(rng.choice(away)) if rng.random() < 0.7 else str(w.home_city[user])
        mule = w.mule()
        for _ in range(size):
            ratio = rng.lognormal(0, 0.3) if subtle else rng.lognormal(np.log(8), 0.6)
            receiver = mule if rng.random() < 0.7 else w.mule()
            rows.append(w.row(when, user, receiver, w.usual[user] * ratio, "p2p", device, city))
            when += pd.Timedelta(seconds=int(rng.integers(60, 600)))
        made += size
    return rows


def _scam_rows(w, n, kind):
    rng, rows = w.rng, []
    txn_type = "collect_request" if kind == "collect" else "qr_scan"
    centre = np.log(3.0 if kind == "collect" else 2.5)
    for _ in range(n):
        user = w.random_user()
        subtle = rng.random() < 0.15
        ratio = rng.lognormal(0, 0.3) if subtle else rng.lognormal(centre, 0.7)
        device = f"X{int(rng.integers(0, 99999)):05d}" if rng.random() < 0.10 else w.home_device(user)
        receiver = w.mule() if rng.random() < 0.92 else int(rng.choice(w.pool[user]))
        rows.append(w.row(w.random_time(), user, receiver, w.usual[user] * ratio, txn_type,
                          device, str(w.home_city[user])))
    return rows


def _fraud_frame(rows):
    f = pd.DataFrame(rows)
    return pd.DataFrame({
        "timestamp": f["timestamp"],
        "sender_id": _fmt("U", f["sender"], 4),
        "receiver_id": _fmt("P", f["receiver"], 5),
        "amount": np.clip(np.round(f["amount"].astype(float), 2), 1, 100_000),
        "txn_type": f["txn_type"],
        "device_id": f["device"],
        "city": f["city"],
    })


def generate(n_users=config.N_USERS, n_transactions=config.N_TRANSACTIONS,
             fraud_rate=config.FRAUD_RATE, n_days=config.N_DAYS, seed=config.SEED):
    rng = np.random.default_rng(seed)
    w = _World(rng, n_users, n_days)

    n_fraud = int(round(n_transactions * fraud_rate))
    n_takeover = int(round(n_fraud * 0.45))
    n_collect = int(round(n_fraud * 0.30))
    n_qr = n_fraud - n_takeover - n_collect

    genuine = _genuine(w, n_transactions - n_fraud)
    fraud = _fraud_frame(_takeover_rows(w, n_takeover) + _scam_rows(w, n_collect, "collect")
                         + _scam_rows(w, n_qr, "qr"))
    genuine["is_fraud"] = 0
    fraud["is_fraud"] = 1

    df = pd.concat([genuine, fraud], ignore_index=True)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(df))])
    return df


def generate_and_save(path=None, **kwargs):
    path = Path(path or config.DATA_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate(**kwargs)
    df.to_csv(path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic UPI transaction dataset.")
    parser.add_argument("--rows", type=int, default=config.N_TRANSACTIONS)
    parser.add_argument("--fraud-rate", type=float, default=config.FRAUD_RATE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", default=str(config.DATA_PATH))
    args = parser.parse_args()
    df = generate_and_save(args.out, n_transactions=args.rows, fraud_rate=args.fraud_rate, seed=args.seed)
    print(f"Wrote {len(df):,} transactions ({int(df['is_fraud'].sum()):,} fraud) to {args.out}")


if __name__ == "__main__":
    main()
