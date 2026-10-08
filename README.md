# CodeSense
Python code review assistant: AST features -> Random Forest bug-proneness score -> per-function reasons.

## Run locally
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python train.py                                     # prints precision/recall/F1/AUC, saves model.joblib
    uvicorn app.main:app --reload                       # open http://localhost:8000

API docs: http://localhost:8000/docs

## Deploy on Render
1. Push this folder to a GitHub repo.
2. Render dashboard -> New -> Blueprint -> pick the repo (it reads render.yaml). Or New -> Web Service with:
   Build `pip install -r requirements.txt && python train.py`, Start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
3. Open the URL Render gives you. Frontend and backend are one service, so no CORS setup is needed.
Free tier sleeps after inactivity; the first request can take ~50s.

## IMPORTANT: replace the placeholder data
`train.py` uses synthetic data so the app runs immediately. For your report, mine real data
(BugsInPy / GitHub bug-fix commits), run `app.features.extract` on the buggy and fixed versions
of each function, save rows as CSV (columns = FEATURES + `label`, 1 = buggy), then:
    pip install pandas && python train.py --csv data.csv
Commit the CSV (or the trained model.joblib and remove it from .gitignore) so Render trains on it.
