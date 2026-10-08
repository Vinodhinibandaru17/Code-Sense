import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .features import extract
from .model import load, predict

app = FastAPI(title="CodeSense API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
clf, baseline = load()
FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")


class AnalyzeRequest(BaseModel):
    code: str = Field(..., max_length=20000)
    threshold: float = Field(0.5, ge=0.05, le=0.95)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    try:
        funcs = extract(req.code)
    except SyntaxError as e:
        raise HTTPException(400, f"Syntax error on line {e.lineno}: {e.msg}")
    results = []
    for f in funcs:
        score, reasons = predict(clf, baseline, f["features"])
        results.append({"name": f["name"], "line": f["line"], "end_line": f["end_line"],
                        "score": round(score, 3), "flagged": score >= req.threshold,
                        "reasons": reasons, "metrics": f["features"]})
    return {"functions": len(results), "flagged": sum(r["flagged"] for r in results),
            "results": results}


@app.get("/")
def index():
    return FileResponse(FRONTEND)
