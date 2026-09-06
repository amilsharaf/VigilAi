"""
Vigil ML pipeline — Isolation Forest anomaly scoring, TF-IDF + geographic
duplicate detection, and an optional gradient-boosting delay-risk model.

Refactored from the original offline batch script into reusable functions
so they can be called live, per-request, by the FastAPI endpoints in
app/main.py instead of run once against a file on disk.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("vigil.pipeline")

REQUIRED_COLUMNS = [
    "work_id",
    "description",
    "state",
    "district",
    "category",
    "sanctioned_cost",
    "sanction_date",
    "expected_duration_days",
    "completion_date",
    "is_complete",
    "has_completion_image",
    "total_paid",
]
# lat/lon are not required — duplicate detection degrades gracefully without them.
OPTIONAL_GEO_COLUMNS = ["lat", "lon"]

DUPLICATE_SIMILARITY_THRESHOLD = 0.80  # cosine similarity above this = likely duplicate
GEO_THRESHOLD_DEG = 0.05  # ~5.5 km — same-district works closer than this are plausibly the same site
MIN_LABELED_FOR_DELAY_MODEL = 40  # don't train a delay model on too little data


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of required columns missing from df (empty if none)."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shared feature engineering used by both the anomaly and delay models."""
    df = df.copy()

    # CSV booleans can arrive as "True"/"False", "TRUE"/"FALSE", 1/0, etc.
    df["is_complete"] = (
        df["is_complete"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    )

    df["sanction_date"] = pd.to_datetime(df["sanction_date"], format="mixed", errors="coerce")
    df["completion_date"] = pd.to_datetime(df["completion_date"], format="mixed", errors="coerce")
    df["actual_duration_days"] = (df["completion_date"] - df["sanction_date"]).dt.days

    cat_cost_median = df.groupby("category")["sanctioned_cost"].transform("median")
    df["cost_ratio_to_category_median"] = df["sanctioned_cost"] / cat_cost_median

    df["duration_ratio_to_expected"] = (
        df["actual_duration_days"] / df["expected_duration_days"]
    ).fillna(-1)  # -1 marks "not yet complete", kept as its own signal, not dropped

    if "total_paid" in df.columns:
        df["paid_fraction_of_sanctioned"] = (df["total_paid"] / df["sanctioned_cost"]).fillna(0)
    else:
        df["paid_fraction_of_sanctioned"] = 0.0

    cat_enc = LabelEncoder()
    df["category_enc"] = cat_enc.fit_transform(df["category"].astype(str))
    district_enc = LabelEncoder()
    df["district_enc"] = district_enc.fit_transform(df["district"].astype(str))

    # District-level spend concentration: how much of the district's total
    # sanctioned spend is tied up in this one work.
    district_total_spend = df.groupby("district")["sanctioned_cost"].transform("sum")
    df["district_spend_concentration"] = df["sanctioned_cost"] / district_total_spend

    return df


def compute_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Unsupervised Isolation Forest anomaly score, trained fresh on this dataset."""
    iso_features = [
        "cost_ratio_to_category_median",
        "duration_ratio_to_expected",
        "paid_fraction_of_sanctioned",
        "district_spend_concentration",
        "category_enc",
        "district_enc",
    ]
    X_iso = df[iso_features].replace([np.inf, -np.inf], 0).fillna(0)

    iso = IsolationForest(n_estimators=300, contamination=0.08, random_state=42)
    iso.fit(X_iso)

    # decision_function: higher = more normal. Flip and rescale to 0-100.
    raw_scores = -iso.decision_function(X_iso)
    score_range = raw_scores.max() - raw_scores.min()
    if score_range == 0:
        # Every row scored identically (e.g. a single-row dataset) — nothing stands out.
        df["ml_anomaly_score"] = 0.0
    else:
        df["ml_anomaly_score"] = (
            (raw_scores - raw_scores.min()) / score_range * 100
        ).round(1)

    return df


def compute_duplicate_clusters(
    df: pd.DataFrame,
    similarity_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
    geo_threshold_deg: float = GEO_THRESHOLD_DEG,
) -> tuple[pd.DataFrame, dict]:
    """TF-IDF + cosine similarity duplicate detection, constrained to same-district
    comparisons (and, when lat/lon are available, geographic proximity)."""
    df["duplicate_cluster_id"] = ""
    meta = {
        "computed": True,
        "geo_constraint_used": False,
        "clusters_found": 0,
        "works_in_clusters": 0,
        "note": None,
    }

    has_geo = "lat" in df.columns and "lon" in df.columns
    meta["geo_constraint_used"] = has_geo
    if not has_geo:
        meta["note"] = (
            "lat/lon columns not provided — duplicate detection used text "
            "similarity only, without the geographic proximity constraint."
        )

    descriptions = df["description"].fillna("")
    if descriptions.str.strip().eq("").all():
        meta["computed"] = False
        meta["note"] = "All descriptions were empty — duplicate detection skipped."
        return df, meta

    try:
        tfidf = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
        # Kept sparse throughout — only small per-district slices are ever
        # densified via cosine_similarity below.
        tfidf_matrix = tfidf.fit_transform(descriptions)
    except ValueError as exc:
        logger.warning("TF-IDF vectorization failed: %s", exc)
        meta["computed"] = False
        meta["note"] = (
            "Description text had no usable vocabulary (e.g. all stop words) — "
            "duplicate detection skipped."
        )
        return df, meta

    cluster_counter = 1
    for _district, group in df.groupby("district"):
        idx = group.index.to_list()
        if len(idx) < 2:
            continue  # nothing to compare a lone work in its district against

        sub_matrix = tfidf_matrix[idx]
        sim = cosine_similarity(sub_matrix)
        np.fill_diagonal(sim, 0)

        visited: set = set()
        for i_local, i_global in enumerate(idx):
            if i_global in visited:
                continue
            matches = []
            for j_local, j_global in enumerate(idx):
                if j_global == i_global or j_global in visited:
                    continue
                if sim[i_local, j_local] < similarity_threshold:
                    continue
                if has_geo:
                    lat_i, lon_i = df.loc[i_global, "lat"], df.loc[i_global, "lon"]
                    lat_j, lon_j = df.loc[j_global, "lat"], df.loc[j_global, "lon"]
                    if pd.isna(lat_i) or pd.isna(lon_i) or pd.isna(lat_j) or pd.isna(lon_j):
                        continue  # can't confirm proximity without both coordinates
                    geo_dist = abs(lat_i - lat_j) + abs(lon_i - lon_j)
                    if geo_dist > geo_threshold_deg:
                        continue  # similar wording but far apart — not a real duplicate
                matches.append(j_global)
            if matches:
                cluster_id = f"DUP{cluster_counter:04d}"
                cluster_counter += 1
                all_members = [i_global] + matches
                for m in all_members:
                    df.loc[m, "duplicate_cluster_id"] = cluster_id
                    visited.add(m)

    meta["clusters_found"] = cluster_counter - 1
    meta["works_in_clusters"] = int((df["duplicate_cluster_id"] != "").sum())
    return df, meta


def compute_delay_risk(
    df: pd.DataFrame, min_labeled: int = MIN_LABELED_FOR_DELAY_MODEL
) -> tuple[pd.DataFrame, dict]:
    """Gradient boosting delay-risk model — only trained if there's enough
    labeled history (completed works with a known actual duration)."""
    meta = {
        "computed": False,
        "reason": None,
        "completed_works_count": 0,
        "test_roc_auc": None,
    }

    completed = df[df["is_complete"] == True].dropna(subset=["actual_duration_days"]).copy()  # noqa: E712
    meta["completed_works_count"] = len(completed)

    if len(completed) < min_labeled:
        meta["reason"] = (
            f"only {len(completed)} completed works with known duration, "
            f"need at least {min_labeled}"
        )
        return df, meta

    completed["was_delayed"] = (
        completed["actual_duration_days"] > completed["expected_duration_days"] * 1.3
    ).astype(int)

    if completed["was_delayed"].nunique() < 2:
        meta["reason"] = "not enough delayed/on-time examples to learn from (only one class present)"
        return df, meta

    # Season of sanction — MPLADS work patterns plausibly differ by monsoon
    # vs. non-monsoon sanctioning, financial-year-end rush, etc.
    completed["sanction_month"] = completed["sanction_date"].dt.month
    df["sanction_month"] = df["sanction_date"].dt.month

    delay_features = [
        "sanctioned_cost",
        "expected_duration_days",
        "category_enc",
        "district_enc",
        "sanction_month",
    ]
    Xd = completed[delay_features]
    yd = completed["was_delayed"]

    Xd_train, Xd_test, yd_train, yd_test = train_test_split(
        Xd, yd, test_size=0.25, random_state=42, stratify=yd
    )
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)
    model.fit(Xd_train, yd_train)
    auc = roc_auc_score(yd_test, model.predict_proba(Xd_test)[:, 1])

    df["delay_risk_score"] = (model.predict_proba(df[delay_features])[:, 1] * 100).round(1)

    meta["computed"] = True
    meta["test_roc_auc"] = round(float(auc), 3)
    return df, meta


def run_ml_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run all three stages and return the original rows plus new ML columns,
    alongside a meta dict describing what was computed and what was skipped."""
    if len(df) == 0:
        raise ValueError("CSV contains no data rows")

    original_df = df.copy()

    work_df = _prepare_features(df)
    work_df = compute_anomaly_scores(work_df)
    work_df, dup_meta = compute_duplicate_clusters(work_df)
    work_df, delay_meta = compute_delay_risk(work_df)

    output_df = original_df.copy()
    output_df["ml_anomaly_score"] = work_df["ml_anomaly_score"]
    output_df["duplicate_cluster_id"] = work_df["duplicate_cluster_id"]
    if delay_meta["computed"]:
        output_df["delay_risk_score"] = work_df["delay_risk_score"]

    meta = {
        "rows_processed": len(output_df),
        "ml_anomaly_score": {"computed": True},
        "duplicate_cluster_id": dup_meta,
        "delay_risk_score": delay_meta,
    }

    logger.info(
        "Pipeline run: %d rows processed, %d anomalies flagged (score>=70), "
        "%d duplicate clusters covering %d works, delay-risk model %s",
        len(output_df),
        int((work_df["ml_anomaly_score"] >= 70).sum()),
        dup_meta["clusters_found"],
        dup_meta["works_in_clusters"],
        "trained" if delay_meta["computed"] else f"skipped ({delay_meta['reason']})",
    )

    return output_df, meta
