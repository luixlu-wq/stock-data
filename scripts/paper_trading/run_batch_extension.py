"""
Batch Paper Trading Extension
==============================
Extends (or re-runs) the paper trading simulation over a full date range and
re-syncs all three Qdrant collections from scratch.

Why batch instead of day-by-day?
  The PaperTradingRunner carries vol-scaling history and PnL history as object
  state. When the daily script creates a fresh runner instance per day, that
  history is lost and kill-switch / vol-scale calculations are inaccurate for
  the first ~60 days. Running all dates inside one runner instance gives
  correct results from day 1.

What this script does
  1. Run full simulation: phase1_predictions.parquet → phase4 output files
     (overwrites phase4_paper_trading_daily.parquet + summary.json)
  2. Clear stock_recommendations, trading_results, performance_metrics in Qdrant
  3. Re-upload every date's recommendations and results to Qdrant
  4. Upload final cumulative performance metrics snapshot
  5. Update paper_trading_progress.json

Usage
  cd C:/Users/luixj/AI/stock-data
  venv\\Scripts\\activate          # Windows
  python scripts/paper_trading/run_batch_extension.py
  python scripts/paper_trading/run_batch_extension.py --start-date 2025-04-01 --end-date 2025-12-31
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_trading.phase4_paper_trading_runner import PaperTradingRunner

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data/processed/phase1_predictions.parquet"
OUTPUT_DIR               = PROJECT_ROOT / "data/processed/phase4"
PROGRESS_FILE            = OUTPUT_DIR / "paper_trading_progress.json"

# ── Qdrant configuration ──────────────────────────────────────────────────────
QDRANT_URL                  = "http://localhost:6333"
COLLECTION_RECOMMENDATIONS  = "stock_recommendations"
COLLECTION_TRADING_RESULTS  = "trading_results"
COLLECTION_PERFORMANCE      = "performance_metrics"

# Strategy constants (STRATEGY_DEFINITION.md v2.0.0)
K              = 38
LONG_EXPOSURE  = 0.65
SHORT_EXPOSURE = 0.35

# ── Logging ───────────────────────────────────────────────────────────────────
log_dir = PROJECT_ROOT / "logs/paper_trading"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_dir / f"batch_extension_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL)


def _ensure_collection(client, name: str, size: int, distance_str: str = "Cosine"):
    from qdrant_client.models import Distance, VectorParams
    distance_map = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}
    distance = distance_map[distance_str]
    try:
        client.get_collection(name)
        log.info("Collection '%s' already exists — will clear it.", name)
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=distance),
        )
        log.info("Created collection '%s'.", name)


def _clear_collection(client, name: str):
    """Delete all points from a collection without dropping it."""
    from qdrant_client.models import Filter
    client.delete(collection_name=name, points_selector=Filter(must=[]))
    log.info("Cleared all points from '%s'.", name)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Run batch simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(start_date: str, end_date: str, predictions_path: Path) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("STEP 1: Running batch paper trading simulation")
    log.info("  Predictions : %s", predictions_path)
    log.info("  Date range  : %s → %s", start_date, end_date)
    log.info("=" * 60)

    runner = PaperTradingRunner(
        predictions_path=str(predictions_path),
        output_dir=str(OUTPUT_DIR),
    )

    results_df, summary = runner.run_backtest_simulation(
        start_date=start_date,
        end_date=end_date,
    )

    log.info(
        "Simulation complete: %d days | Sharpe(scaled)=%.2f | MaxDD=%.2f%%",
        summary["num_days"],
        summary["sharpe_scaled"],
        summary["max_dd"] * 100,
    )
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build recommendations per date
# ─────────────────────────────────────────────────────────────────────────────

def _build_recommendations_for_date(date: str, pred_df: pd.DataFrame) -> list[dict]:
    """Apply S2_FilterNegative logic to predictions and return flat list of dicts."""
    day = pred_df[pred_df["date"] == date].copy()
    if day.empty:
        return []

    day_sorted = day.sort_values("y_pred_reg", ascending=False)

    longs = day_sorted.head(K)
    long_weight = LONG_EXPOSURE / len(longs)

    short_candidates = day_sorted.tail(K)
    shorts = short_candidates[short_candidates["y_pred_reg"] < 0]
    short_weight = -SHORT_EXPOSURE / len(shorts) if len(shorts) > 0 else 0.0

    records = []
    for _, row in longs.iterrows():
        records.append({
            "ticker":         row["ticker"],
            "position":       "LONG",
            "weight":         float(long_weight),
            "prediction":     float(row["y_pred_reg"]),
            "actual_return":  float(row["y_true_reg"]) if not pd.isna(row.get("y_true_reg")) else None,
            "close_price":    float(row["close"]) if not pd.isna(row.get("close")) else None,
            "date":           date,
        })

    for _, row in shorts.iterrows():
        records.append({
            "ticker":         row["ticker"],
            "position":       "SHORT",
            "weight":         float(short_weight),
            "prediction":     float(row["y_pred_reg"]),
            "actual_return":  float(row["y_true_reg"]) if not pd.isna(row.get("y_true_reg")) else None,
            "close_price":    float(row["close"]) if not pd.isna(row.get("close")) else None,
            "date":           date,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Sync Qdrant
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v, default: float = 0.0) -> float:
    """Convert to float, replacing NaN/Inf with default (Qdrant rejects non-finite values)."""
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def sync_qdrant(results_df: pd.DataFrame, pred_df: pd.DataFrame):
    from qdrant_client.models import PointStruct

    log.info("=" * 60)
    log.info("STEP 2: Syncing Qdrant collections")
    log.info("=" * 60)

    client = _load_qdrant_client()

    # Ensure collections exist
    _ensure_collection(client, COLLECTION_RECOMMENDATIONS, size=10)
    _ensure_collection(client, COLLECTION_TRADING_RESULTS, size=5)
    _ensure_collection(client, COLLECTION_PERFORMANCE,     size=8)

    # Clear stale data
    _clear_collection(client, COLLECTION_RECOMMENDATIONS)
    _clear_collection(client, COLLECTION_TRADING_RESULTS)
    _clear_collection(client, COLLECTION_PERFORMANCE)

    dates = sorted(str(d)[:10] for d in results_df["date"].unique())
    total_reco_points = 0
    total_result_points = 0

    # ── Upload per-date data ──────────────────────────────────────────────
    # Use a deterministic integer ID: encode date as YYYYMMDD * 10^6 + offset
    for date_str in dates:
        date_int = int(date_str.replace("-", ""))  # e.g. 20250401

        # --- Recommendations ---
        reco_records = _build_recommendations_for_date(date_str, pred_df)
        if reco_records:
            reco_points = []
            for idx, rec in enumerate(reco_records):
                point_id = date_int * 10_000 + idx  # unique per date+position
                vector = [
                    _safe_float(rec["weight"]),
                    _safe_float(rec["prediction"]),
                    _safe_float(rec["actual_return"]),
                    _safe_float(rec["close_price"], 1.0) / 1000.0,
                    1.0 if rec["position"] == "LONG" else -1.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ]
                reco_points.append(PointStruct(id=point_id, vector=vector, payload=rec))

            client.upsert(collection_name=COLLECTION_RECOMMENDATIONS, points=reco_points)
            total_reco_points += len(reco_points)

        # --- Trading result ---
        day_row = results_df[results_df["date"] == date_str]
        if not day_row.empty:
            r = day_row.iloc[0].to_dict()
            result_vector = [
                _safe_float(r.get("pnl_net",      0.0)),
                _safe_float(r.get("pnl_scaled",   0.0)),
                _safe_float(r.get("turnover",     0.0)),
                _safe_float(r.get("vol_scale",    1.0)),
                1.0 if r.get("ks_triggered") else 0.0,
            ]
            result_payload = {k: (bool(v)        if isinstance(v, (bool, np.bool_))     else
                                  _safe_float(v) if isinstance(v, (float, np.floating)) else
                                  int(v)         if isinstance(v, (int, np.integer))    else v)
                              for k, v in r.items()}
            result_payload["historical_date"] = date_str

            result_point_id = date_int * 10_000 + 9_999  # one result per date
            client.upsert(
                collection_name=COLLECTION_TRADING_RESULTS,
                points=[PointStruct(id=result_point_id, vector=result_vector, payload=result_payload)],
            )
            total_result_points += 1

    log.info("Uploaded %d recommendation points.", total_reco_points)
    log.info("Uploaded %d trading result points.", total_result_points)

    # ── Upload performance snapshot ───────────────────────────────────────
    summary_path = OUTPUT_DIR / "phase4_paper_trading_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    perf_vector = [
        _safe_float(summary.get("sharpe_scaled",        0)),
        _safe_float(summary.get("annual_return_scaled",  0)),
        _safe_float(summary.get("vol_scaled",            0)),
        _safe_float(summary.get("max_dd",                0)),
        _safe_float(summary.get("long_sharpe",           0)),
        _safe_float(summary.get("short_sharpe",          0)),
        _safe_float(summary.get("avg_turnover",          0)),
        _safe_float(summary.get("ks_pct",                0)) / 100.0,
    ]

    perf_point_id = int(datetime.now().timestamp())
    client.upsert(
        collection_name=COLLECTION_PERFORMANCE,
        points=[PointStruct(
            id=perf_point_id,
            vector=perf_vector,
            payload={
                **summary,
                "calendar_date": datetime.now().strftime("%Y-%m-%d"),
                "saved_at":      datetime.now().isoformat(),
                "deployment_ready": (
                    summary.get("sharpe_scaled", 0) > 1.0
                    and summary.get("max_dd", 0) > -0.10
                ),
            },
        )],
    )
    log.info("Uploaded performance metrics snapshot.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Update progress file
# ─────────────────────────────────────────────────────────────────────────────

def update_progress(results_df: pd.DataFrame, start_date: str):
    dates = sorted(str(d)[:10] for d in results_df["date"].unique())
    last_date = dates[-1]

    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
    else:
        progress = {
            "start_calendar_date":  datetime.now().strftime("%Y-%m-%d"),
            "start_historical_date": start_date,
        }

    progress["start_historical_date"] = start_date
    progress["last_historical_date"]  = last_date
    progress["last_calendar_date"]    = datetime.now().strftime("%Y-%m-%d")
    progress["days_completed"]        = len(dates)

    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

    log.info(
        "Progress updated: %d days completed | last historical date: %s",
        len(dates), last_date,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch paper trading extension")
    parser.add_argument("--start-date", default="2025-04-01",
                        help="Start date YYYY-MM-DD (default: 2025-04-01)")
    parser.add_argument("--end-date",   default="2025-12-31",
                        help="End date YYYY-MM-DD (default: 2025-12-31)")
    parser.add_argument("--predictions", default=None,
                        help="Path to predictions parquet file "
                             "(default: data/processed/phase1_predictions.parquet)")
    parser.add_argument("--skip-qdrant", action="store_true",
                        help="Skip Qdrant sync (only regenerate local files)")
    args = parser.parse_args()

    predictions_path = Path(args.predictions) if args.predictions else DEFAULT_PREDICTIONS_PATH

    log.info("=" * 60)
    log.info("BATCH PAPER TRADING EXTENSION")
    log.info("  Range      : %s → %s", args.start_date, args.end_date)
    log.info("  Predictions: %s", predictions_path)
    log.info("=" * 60)

    # Load predictions once (shared by runner + recommendation builder)
    log.info("Loading predictions from %s", predictions_path)
    pred_df = pd.read_parquet(predictions_path)
    # Normalise to plain strings so comparisons work regardless of parquet storage type
    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.strftime("%Y-%m-%d")

    available_min = pred_df["date"].min()
    available_max = pred_df["date"].max()
    log.info("Predictions available: %s → %s (%d dates, %d tickers)",
             available_min, available_max,
             pred_df["date"].nunique(), pred_df["ticker"].nunique())

    if args.end_date > available_max:
        log.warning(
            "Requested end date %s is beyond available predictions (%s). "
            "Capping at %s.",
            args.end_date, available_max, available_max,
        )
        args.end_date = available_max

    # Step 1: Run simulation
    results_df = run_simulation(args.start_date, args.end_date, predictions_path)

    # Step 2: Sync Qdrant
    if not args.skip_qdrant:
        sync_qdrant(results_df, pred_df)
    else:
        log.info("Skipping Qdrant sync (--skip-qdrant flag set).")

    # Step 3: Update progress
    update_progress(results_df, args.start_date)

    log.info("=" * 60)
    log.info("ALL DONE — paper trading extended to %s", args.end_date)
    log.info("Output files:")
    log.info("  Predictions : %s", predictions_path)
    log.info("  Daily       : %s", OUTPUT_DIR / "phase4_paper_trading_daily.parquet")
    log.info("  Summary     : %s", OUTPUT_DIR / "phase4_paper_trading_summary.json")
    log.info("  Progress    : %s", PROGRESS_FILE)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
