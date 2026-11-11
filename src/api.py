"""Loopback-only dashboard and local analysis service. Run via scripts/start.sh."""

import json
import os
import tempfile
import threading
from pathlib import Path
import joblib
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from .ingest import PostDataIngestor
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from .pipeline import run_pipeline, score_post

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD = 5 * 1024 * 1024
RUN_DIR = Path(os.environ.get("CAMPUS_REPORT_DIR", ROOT / "reports/latest"))
app = FastAPI(title="CampusTrendsML local analysis")
lock = threading.Lock()
state = {"artifact": None}


def get_artifact():
    if state["artifact"] is None:
        path = RUN_DIR / "model.joblib"
        if not path.exists():
            raise HTTPException(503, "Run the pipeline before starting the dashboard.")
        # Only load artifacts created by this local pipeline, never uploaded pickles.
        state["artifact"] = joblib.load(path)
    return state["artifact"]


@app.middleware("http")
async def local_only(request, call_next):
    if request.url.path.startswith("/api"):
        host = request.headers.get("host", "").split(":")[0]
        origin = request.headers.get("origin")
        if host not in ("127.0.0.1", "localhost", "testserver"):
            return JSONResponse(
                {"detail": "Use the local dashboard address."}, status_code=403
            )
        if origin and origin.rstrip("/") not in (
            f"http://{request.headers.get('host')}",
            "http://127.0.0.1:8030",
            "http://localhost:8030",
        ):
            return JSONResponse(
                {"detail": "Cross-origin API requests are disabled."}, status_code=403
            )
    return await call_next(request)


@app.get("/api/report")
def report():
    return get_artifact()["report"]


class Draft(BaseModel):
    text: str = Field(min_length=5, max_length=20000)
    timestamp: str


@app.post("/api/score")
def score(draft: Draft):
    try:
        return score_post(get_artifact(), draft.text, draft.timestamp)
    except (ValueError, OverflowError, TypeError) as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), calendar: str = Form("[]")):
    if Path(file.filename or "").suffix.lower() not in (".csv", ".json"):
        raise HTTPException(422, "Choose a CSV or JSON file.")
    content = await file.read(MAX_UPLOAD + 1)
    if len(content) > MAX_UPLOAD:
        raise HTTPException(413, "Uploads must be smaller than 5 MB.")
    if not lock.acquire(blocking=False):
        raise HTTPException(
            409, "An analysis is already running. Try again when it finishes."
        )

    def process():
        with tempfile.TemporaryDirectory(prefix="campustrends-") as directory:
            path = Path(directory) / ("posts" + Path(file.filename).suffix.lower())
            path.write_bytes(content)
            reader = PostDataIngestor()
            raw = len(
                reader.load_csv(path)
                if path.suffix == ".csv"
                else reader.load_json(path)
            )
            if raw > 10000:
                raise ValueError("Use the CLI for more than 10,000 rows.")
            result = run_pipeline(
                path,
                Path(directory) / "output",
                calendar=json.loads(calendar),
                provenance="user-supplied",
            )
            state["artifact"] = joblib.load(Path(directory) / "output/model.joblib")
            return result

    try:
        return await run_in_threadpool(process)
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError) as error:
        raise HTTPException(422, str(error)) from error
    finally:
        lock.release()


@app.post("/api/reset")
def reset():
    state["artifact"] = None
    return get_artifact()["report"]


if (ROOT / "dist").exists():
    app.mount("/", StaticFiles(directory=ROOT / "dist", html=True), name="dashboard")
