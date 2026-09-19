"""
Vigil ML Service — FastAPI wrapper around the Vigil ML pipeline.

Called by the Lovable frontend via fetch(); this repo is backend-only.
No uploaded data is persisted to disk — each request is scored in memory
and the enriched result is returned directly in the response.
"""

import io
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import run_ml_pipeline, validate_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vigil.api")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

app = FastAPI(title="Vigil ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vigil-govin.lovable.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
async def score(file: UploadFile = File(...)):
    # Reject oversized uploads early when the client sends Content-Length.
    content_length = file.size
    if content_length is not None and content_length > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large — max upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large — max upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
        )

    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no data")
    except Exception:
        logger.exception("Failed to parse uploaded CSV")
        raise HTTPException(status_code=400, detail="Could not parse the uploaded file as CSV")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV contains no data rows")

    missing = validate_columns(df)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {', '.join(missing)}",
        )

    try:
        result_df, meta = run_ml_pipeline(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Unexpected error while scoring uploaded CSV")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while processing the file"
        )

    # NaN/NaT/inf aren't valid JSON — normalize to null before serializing.
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    result_df = result_df.astype(object).where(pd.notnull(result_df), None)

    return {"data": result_df.to_dict(orient="records"), "meta": meta}
