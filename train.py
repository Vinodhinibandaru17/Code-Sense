"""python train.py            -> trains on placeholder data
   python train.py --csv f.csv -> trains on your data (columns = FEATURES + `label` 0/1)"""
import sys, numpy as np
from app.features import FEATURES
from app.model import train, synthetic_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

if "--csv" in sys.argv:
    import pandas as pd  # pip install pandas (local only)
    df = pd.read_csv(sys.argv[sys.argv.index("--csv") + 1])
    X, y = df[FEATURES].values.astype(float), df["label"].values
else:
    X, y = synthetic_dataset()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
clf, _ = train(Xtr, ytr)
pred, prob = clf.predict(Xte), clf.predict_proba(Xte)[:, 1]
p, r, f, _ = precision_recall_fscore_support(yte, pred, average="binary")
print(f"precision={p:.3f} recall={r:.3f} f1={f:.3f} auc={roc_auc_score(yte, prob):.3f}")
