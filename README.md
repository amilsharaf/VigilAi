# vigil-ml-service

FastAPI backend that exposes the Vigil ML enrichment pipeline (Isolation
Forest anomaly scoring, TF-IDF + geographic-proximity duplicate detection,
and an optional gradient-boosting delay-risk model) as live HTTP endpoints.

Backend only — no frontend/UI here. It's called by the separate Lovable
frontend via `fetch()`. No uploaded data is persisted to disk; each request
is processed in memory and the enriched result is returned in the response.

## Project layout

```
app/
  main.py      FastAPI app and endpoints
  pipeline.py  Refactored ML pipeline (Isolation Forest, TF-IDF/geo duplicates, delay-risk model)
test_data/
  sample_works.csv   16-row synthetic dataset for testing /score
requirements.txt
Procfile
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate   # on Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The service is then available at `http://localhost:8000`. Interactive docs
at `http://localhost:8000/docs`.

Test it with the included sample CSV:

```bash
curl -F "file=@test_data/sample_works.csv" http://localhost:8000/score
```

## Deploy to Render (free-tier web service)

1. Push this repo to GitHub.
2. In the Render dashboard: **New > Web Service**, connect the repo.
3. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
     (or leave blank — Render will pick up the `Procfile`)
   - **Instance Type**: Free
4. Deploy. Render assigns a public URL like `https://vigil-ml-service.onrender.com`.
5. Once the Lovable app's deployed URL is known, tighten CORS in
   `app/main.py` (`allow_origins=["*"]` → the specific frontend origin)
   and redeploy.

Note: free-tier Render services spin down after inactivity, so the first
request after idle time will be slow (cold start) — expected on this tier.

## Endpoints

### `GET /health`

Liveness check.

```json
{ "status": "ok" }
```

### `POST /score`

Accepts a CSV file upload (`multipart/form-data`, field name `file`, max
10MB) with columns:

```
work_id, description, state, district, category, sanctioned_cost,
sanction_date, expected_duration_days, completion_date, is_complete,
has_completion_image, total_paid, lat, lon
```

`lat`/`lon` are optional — if omitted, duplicate detection still runs on
text similarity alone (noted in the response `meta`).

Returns the original rows plus new ML-derived fields:

- `ml_anomaly_score` — 0-100, always computed
- `duplicate_cluster_id` — empty string if the row isn't part of a cluster
- `delay_risk_score` — only present if the pipeline had ≥40 completed
  works with known duration to train on; omitted entirely otherwise

```json
{
  "data": [
    {
      "work_id": "W001",
      "description": "Construction of concrete road at Shivaji Nagar",
      "...": "...",
      "ml_anomaly_score": 12.4,
      "duplicate_cluster_id": "DUP0001",
      "delay_risk_score": 38.2
    }
  ],
  "meta": {
    "rows_processed": 16,
    "ml_anomaly_score": { "computed": true },
    "duplicate_cluster_id": {
      "computed": true,
      "geo_constraint_used": true,
      "clusters_found": 1,
      "works_in_clusters": 2,
      "note": null
    },
    "delay_risk_score": {
      "computed": false,
      "reason": "only 12 completed works with known duration, need at least 40",
      "completed_works_count": 12,
      "test_roc_auc": null
    }
  }
}
```

A CSV missing required columns returns `400` naming them:

```json
{ "detail": "CSV is missing required columns: sanction_date, category" }
```

## Calling /score from the Lovable frontend

```js
async function scoreWorksCsv(file) {
  const formData = new FormData();
  formData.append("file", file); // file: File object from an <input type="file">

  const response = await fetch("https://vigil-ml-service.onrender.com/score", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Scoring failed");
  }

  const { data, meta } = await response.json();
  return { data, meta };
}
```

Do not set a `Content-Type` header manually — the browser sets the correct
`multipart/form-data` boundary automatically when you pass a `FormData` body.
