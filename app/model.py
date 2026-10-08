"""Classifier training/loading + occlusion-based explanations."""
import os, joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier
from .features import FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")

REASONS = {
    "bare_except": ("Bare `except:` swallows every error, including KeyboardInterrupt.",
                    "Catch specific exceptions and log the error."),
    "broad_except": ("`except Exception` is very broad and can hide real bugs.",
                     "Catch the specific exceptions you expect."),
    "except_pass": ("Exception handler only does `pass`, so failures are silent.",
                    "Log or re-raise the exception instead of ignoring it."),
    "mutable_default": ("Mutable default argument is shared between calls.",
                        "Use `None` as default and create the list/dict inside the function."),
    "open_no_with": ("File opened without a `with` block may never be closed (resource leak).",
                     "Use `with open(...) as f:` so the file closes automatically."),
    "cyclomatic": ("High cyclomatic complexity: many branches make bugs likely.",
                   "Split the function into smaller helpers."),
    "max_nesting": ("Deeply nested control flow is hard to reason about.",
                    "Use early returns or extract inner blocks."),
    "loc": ("Very long function.", "Break it into smaller, single-purpose functions."),
    "n_params": ("Too many parameters.", "Group related parameters into an object."),
    "n_try": ("Multiple try blocks in one function.", "Isolate error handling in helpers."),
    "uses_global": ("Uses global/nonlocal state.", "Pass values in and return them instead."),
    "n_returns": ("Many return points.", "Reduce exit points for clearer flow."),
}


def synthetic_dataset(n=6000, seed=0):
    """PLACEHOLDER data so the app runs out of the box. Replace with mined bug-fix data
    (see README) via `python train.py --csv your_data.csv`."""
    r = np.random.default_rng(seed)
    X = np.column_stack([
        r.integers(2, 90, n), r.integers(0, 9, n), r.integers(1, 25, n), r.integers(0, 6, n),
        r.poisson(0.5, n), r.binomial(1, 0.12, n), r.binomial(1, 0.15, n), r.binomial(1, 0.1, n),
        r.binomial(1, 0.07, n), r.binomial(1, 0.15, n), r.integers(0, 8, n), r.binomial(1, 0.06, n)])
    z = (-3.2 + 0.03 * X[:, 0] + 0.12 * X[:, 1] + 0.14 * X[:, 2] + 0.35 * X[:, 3] + 1.6 * X[:, 5]
         + 0.9 * X[:, 6] + 1.2 * X[:, 7] + 1.4 * X[:, 8] + 1.1 * X[:, 9] + 0.8 * X[:, 11]
         + r.normal(0, 0.8, n))
    return X.astype(float), (z > 0).astype(int)


def train(X, y, path=MODEL_PATH):
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5,
                                 random_state=42, n_jobs=-1).fit(X, y)
    baseline = np.median(X[y == 0], axis=0)  # what a "clean" function looks like
    joblib.dump({"clf": clf, "baseline": baseline}, path)
    return clf, baseline


def load():
    if not os.path.exists(MODEL_PATH):
        train(*synthetic_dataset())
    d = joblib.load(MODEL_PATH)
    return d["clf"], d["baseline"]


def predict(clf, baseline, feats: dict, top_k=3):
    x = np.array([[feats[f] for f in FEATURES]], dtype=float)
    p = float(clf.predict_proba(x)[0, 1])
    # Occlusion: how much does the score drop if this feature looked "clean"?
    rows = np.repeat(x, len(FEATURES), axis=0)
    for i in range(len(FEATURES)):
        rows[i, i] = baseline[i]
    drops = p - clf.predict_proba(rows)[:, 1]
    order = np.argsort(-drops)
    reasons = [{"feature": FEATURES[i], "impact": round(float(drops[i]), 3),
                "text": REASONS[FEATURES[i]][0], "fix": REASONS[FEATURES[i]][1]}
               for i in order[:top_k] if drops[i] > 0.01]
    return p, reasons
