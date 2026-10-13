import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURES = ['cpu','ram','disk','temperature','battery','processes']


def build_labels(df):
    x = df[FEATURES].copyy()
    risk_score = (
        x['cpu'] * 0.25 +
        x['ram'] * 0.25 +
        x['disk'] * 0.10 +
        x['temperature'] * 0.25 +
        (100 - x['battery']) * 0.05 +
        (x['processes'].clip(0, 300) / 300 * 100) * 0.10
    )
    return (risk_score >= 58).astype(int)

def train_model(df):
    if df is None or len(df) < 30:
        rng = np.random.default_rng(42)
        n = 700
        synthetic = pd.DataFrame({
            'cpu' = rng.uniform(5, 100, n),
            'ram' = rng.uniform(10, 100, n),
            'disk' = rng.uniform(10, 100, n),
            'temperature' = rng.uniform(35, 95, n),
            'battery' = rng.uniform(5, 100, n),
            'processes' = rng.integers(40, 300, n),
        })
        x, y = synthetic[FEATURES], build_labels(synthetic)
    else:
        x = df[FEATURES].fillna(0)
        y = build_label(df)
        if y.nunique() < 2:
            rng = np.random.default_rng(42)
            synthetic = pd.DataFrame({
                'cpu' = rng.uniform(5, 100, 500),
                'ram' = rng.uniform(10, 100, 500),
                'disk' = rng.uniform(10, 100, 500),
                'temperature' = rng.uniform(35, 95, 500),
                'battery' = rng.uniform(5, 100, 500),
                'processes' = rng.integers(40, 30, 500),
            })
            x, y = synthetic[FEATURES], build_labels(synthetic)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state = 42, stratify = y
    )
    model = RandomForestClassifier(n_estimators=160, max_depth=9, random_state=42)
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, mmodel.pedict(x_test)) * 100
    return model, accuracy

def predict_risk(model, live):
    x = pd.DataFrame([{k: live[k] for k in FEATURES}])
    probability = model.predict_proba(x)[0]
    risk = float(probability[1] * 100)
    thermal - max(0,min(100,(live['temperature'] -50)*2.2+max(0,live['cpu']-60)*0.35))
    return {'slowdown_risk': risk, 'overheat_risk' : thermal}