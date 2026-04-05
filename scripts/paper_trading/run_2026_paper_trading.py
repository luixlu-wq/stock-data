"""
2026 Paper Trading with Live Yahoo Finance Data
================================================
Downloads actual 2026 market data from Yahoo Finance, generates LSTM
predictions for Jan 2 – Feb 18, 2026, then runs a combined 2025+2026
simulation for correct vol-scaling continuity.

Steps:
  1. Download OHLCV from Yahoo Finance (2025-09-01 → today+1)
  2. Compute 14 core features via SimplifiedStockPreprocessor
  3. Load frozen LSTMRegressor (phase2a_temp0.05)
  4. Generate y_pred_reg + y_true_reg for each 2026 trading day
  5. Save to data/processed/phase4/predictions_2026.parquet
  6. Combine with phase1_predictions.parquet (2025 data)
  7. Run full simulation 2025-04-01 → sim-end
  8. Sync Qdrant (clears and re-uploads recommendations, results, performance, and daily strategies)
  9. Update paper_trading_progress.json

Caching:
  OHLCV data is cached in data/processed/phase4/ohlcv_2026_cache.parquet and
  reused automatically on subsequent runs (no re-download unless stale or forced).

Usage:
  cd C:/Users/luixj/AI/stock-data
  python scripts/paper_trading/run_2026_paper_trading.py            # normal run (uses cache if fresh)
  python scripts/paper_trading/run_2026_paper_trading.py --skip-qdrant        # skip Qdrant sync
  python scripts/paper_trading/run_2026_paper_trading.py --force-download     # force fresh download
  python scripts/paper_trading/run_2026_paper_trading.py --skip-download      # skip all download+inference steps
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import ConfigLoader
from src.data.preprocessor_v2 import SimplifiedStockPreprocessor
from src.models.lstm_model import LSTMRegressor
from scripts.paper_trading.phase4_paper_trading_runner import PaperTradingRunner
from scripts.paper_trading.strategy_planner import (
    build_recommendations_for_date,
    build_trade_strategy_payload,
    strategy_payload_to_vector,
)

# =============================================================================
# Paths
# =============================================================================
MODEL_PATH     = PROJECT_ROOT / "models/checkpoints/lstm_phase2a_temp0.05_best.pth"
PRED_2025_PATH = PROJECT_ROOT / "data/processed/phase1_predictions.parquet"
PRED_2026_PATH  = PROJECT_ROOT / "data/processed/phase4/predictions_2026.parquet"
PRED_ALL_PATH   = PROJECT_ROOT / "data/processed/phase4/predictions_combined.parquet"
OHLCV_CACHE_PATH = PROJECT_ROOT / "data/processed/phase4/ohlcv_2026_cache.parquet"
OUTPUT_DIR      = PROJECT_ROOT / "data/processed/phase4"
PROGRESS_FILE  = OUTPUT_DIR / "paper_trading_progress.json"
CONFIG_PATH    = PROJECT_ROOT / "config/config.yaml"

# =============================================================================
# Constants
# =============================================================================
SEQUENCE_LENGTH = 90
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

FEATURE_COLS = [
    'ret_1d', 'ret_5d', 'ret_20d',
    'vol_10d', 'vol_20d',
    'hl_range', 'oc_gap',
    'sma_10_dist', 'sma_20_dist',
    'log_volume', 'volume_change',
    'market_return', 'vs_market', 'market_correlation',
]

SIM_START_2025 = "2025-04-01"    # Start of full combined simulation
SIM_START_2026 = "2026-01-02"    # First 2026 trading day
DEFAULT_SIM_END = "auto"         # Resolve to yesterday at runtime

# Yahoo Finance window: need 90-day lookback before SIM_START_2026.
# 2025-09-01 gives ~86 trading days (~120 calendar days) of lookback.
YF_START = "2025-09-01"
# end is exclusive in yfinance; use today+1 to include today's close
# (needed to compute next-day return for 2026-02-18)
YF_END   = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

# Qdrant
QDRANT_URL                 = "http://localhost:6333"
COLLECTION_RECOMMENDATIONS = "stock_recommendations"
COLLECTION_TRADING_RESULTS = "trading_results"
COLLECTION_PERFORMANCE     = "performance_metrics"
COLLECTION_STRATEGIES      = "daily_trade_strategies"
K             = 38
LONG_EXPOSURE  = 0.65
SHORT_EXPOSURE = 0.35

# =============================================================================
# Logging
# =============================================================================
log_dir = PROJECT_ROOT / "logs/paper_trading"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(
            log_dir / f"run_2026_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def resolve_sim_end(raw_sim_end: str) -> str:
    """Resolve sim-end; 'auto' means yesterday."""
    if str(raw_sim_end).lower() == "auto":
        return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    return pd.to_datetime(raw_sim_end).strftime('%Y-%m-%d')


# =============================================================================
# Step 1: Download OHLCV from Yahoo Finance
# =============================================================================

def download_ohlcv(tickers: list, required_end_date: str, force: bool = False) -> pd.DataFrame:
    """
    Download OHLCV data for all tickers from YF_START to YF_END.

    Results are cached to OHLCV_CACHE_PATH and reused automatically.
    Cache is considered fresh when it contains data through required_end_date
    (i.e., all data needed for the requested simulation window is present).
    Pass force=True to always re-download.
    """
    # ── Cache check ───────────────────────────────────────────────────────────
    if not force and OHLCV_CACHE_PATH.exists():
        cached = pd.read_parquet(OHLCV_CACHE_PATH)
        cached['date'] = pd.to_datetime(cached['date']).dt.tz_localize(None)
        cached_max = cached['date'].max().strftime('%Y-%m-%d')

        if cached_max >= required_end_date:
            log.info("=" * 60)
            log.info("STEP 1: Using cached OHLCV data (up to %s)", cached_max)
            log.info("  Cache : %s", OHLCV_CACHE_PATH)
            log.info("  Tip   : use --force-download to refresh")
            log.info("=" * 60)
            return cached
        else:
            log.info(
                "OHLCV cache is stale (max date %s < SIM_END %s) — re-downloading",
                cached_max, required_end_date,
            )

    log.info("=" * 60)
    log.info(f"STEP 1: Downloading OHLCV data")
    log.info(f"  Range  : {YF_START} -> {YF_END}")
    log.info(f"  Tickers: {len(tickers)}")
    log.info("=" * 60)

    all_data = []
    failed = []

    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            log.info(f"  Progress: {i+1}/{len(tickers)} tickers downloaded")
        try:
            raw = yf.download(
                ticker,
                start=YF_START,
                end=YF_END,
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                log.warning(f"  No data for {ticker}")
                failed.append(ticker)
                continue

            raw = raw.reset_index()

            # Flatten multi-level columns (yfinance >= 0.2 returns MultiIndex)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]

            # Normalize to lowercase
            raw.columns = [str(c).lower().strip() for c in raw.columns]

            # With auto_adjust=True the column is 'close' (adjusted); 'adj close' not present
            rename_map = {'adj close': 'close', 'adj_close': 'close'}
            raw = raw.rename(columns=rename_map)

            required = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in raw.columns]
            if missing:
                log.warning(f"  {ticker}: missing columns {missing} — skipping")
                failed.append(ticker)
                continue

            raw['ticker'] = ticker
            all_data.append(raw[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']])

        except Exception as e:
            log.error(f"  Error downloading {ticker}: {e}")
            failed.append(ticker)

    if failed:
        log.warning(f"Failed tickers ({len(failed)}): {failed[:10]}...")

    if not all_data:
        raise ValueError("No data downloaded for any ticker!")

    df = pd.concat(all_data, ignore_index=True)

    # Strip timezone
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

    log.info(
        f"Downloaded {len(df)} rows | {df['ticker'].nunique()} tickers | "
        f"{df['date'].min().date()} -> {df['date'].max().date()}"
    )

    # Save to cache
    OHLCV_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OHLCV_CACHE_PATH, index=False)
    log.info(f"OHLCV data cached to {OHLCV_CACHE_PATH}")

    return df


# =============================================================================
# Step 2: Compute features
# =============================================================================

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply 14-feature engineering from preprocessor_v2."""
    log.info("=" * 60)
    log.info("STEP 2: Computing 14 core features")
    log.info("=" * 60)

    config = ConfigLoader(str(CONFIG_PATH))
    preprocessor = SimplifiedStockPreprocessor(config)
    features_df = preprocessor.calculate_core_features(df)

    log.info(f"Features: {len(features_df)} rows, {features_df['ticker'].nunique()} tickers")
    return features_df


# =============================================================================
# Step 3: Load model
# =============================================================================

def load_model() -> LSTMRegressor:
    """Load the frozen production LSTMRegressor (phase2a_temp0.05)."""
    log.info("=" * 60)
    log.info(f"STEP 3: Loading model from {MODEL_PATH}")
    log.info("=" * 60)

    # Architecture confirmed from checkpoint state_dict shapes:
    #   lstm.weight_ih_l0: [768, 14]  → 768 = 4 * 192  → hidden_size=192
    #   fc.0.weight:       [64, 192]  → confirms hidden_size=192
    model = LSTMRegressor(
        input_size=14,
        hidden_size=192,
        num_layers=2,
        dropout=0.2,
    )

    checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
        log.info("Model loaded (strict=True)")
    except RuntimeError as e:
        log.warning(f"Strict load failed: {e}")
        model.load_state_dict(state_dict, strict=False)
        log.info("Model loaded (strict=False — some weights may differ)")

    model.to(DEVICE)
    model.eval()
    log.info(f"Model ready on {DEVICE}")
    return model


# =============================================================================
# Step 4: Generate predictions for 2026 trading days
# =============================================================================

def generate_predictions(
    features_df: pd.DataFrame,
    model: LSTMRegressor,
    sim_start: str,
    sim_end: str,
) -> pd.DataFrame:
    """
    Run LSTM inference for each trading day in [sim_start, sim_end].

    For each date D and ticker:
      - Extracts the 90-day feature sequence ending at D
      - Produces y_pred_reg  = LSTM output
      - Produces y_true_reg  = ret_1d of D+1 (actual next-day log return)
      - Records close price at D
    """
    log.info("=" * 60)
    log.info(f"STEP 4: Generating predictions {sim_start} -> {sim_end}")
    log.info("=" * 60)

    # Convert dates to datetime for comparison
    features_df = features_df.copy()
    features_df['date'] = pd.to_datetime(features_df['date']).dt.tz_localize(None)

    all_dates = sorted(features_df['date'].unique())
    sim_start_dt = pd.to_datetime(sim_start)
    sim_end_dt   = pd.to_datetime(sim_end)
    trading_days = [d for d in all_dates if sim_start_dt <= d <= sim_end_dt]

    log.info(f"Trading days in simulation range: {len(trading_days)}")
    if not trading_days:
        raise ValueError(f"No trading days found between {sim_start} and {sim_end}")

    # Build sorted date list and index for next-day return lookups
    date_list    = all_dates            # already sorted
    date_to_idx  = {d: i for i, d in enumerate(date_list)}

    # Group by ticker for fast repeated access
    ticker_groups = {
        ticker: group.sort_values('date').reset_index(drop=True)
        for ticker, group in features_df.groupby('ticker')
    }
    tickers = list(ticker_groups.keys())

    records = []
    for date in trading_days:
        date_str = date.strftime('%Y-%m-%d')
        day_preds = 0

        for ticker in tickers:
            tdf = ticker_groups[ticker]

            # Sequence of up to SEQUENCE_LENGTH rows ending on this date (inclusive)
            seq_mask = tdf['date'] <= date
            seq_data = tdf[seq_mask].tail(SEQUENCE_LENGTH)

            if len(seq_data) < SEQUENCE_LENGTH:
                continue

            # Feature matrix
            missing_feat = [c for c in FEATURE_COLS if c not in seq_data.columns]
            if missing_feat:
                continue

            X = seq_data[FEATURE_COLS].values
            if np.isnan(X).any():
                continue

            # LSTM inference
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(DEVICE)  # (1, 90, 14)
            with torch.no_grad():
                y_pred = float(model(X_tensor).cpu().numpy()[0, 0])

            # Close price at prediction date
            date_rows = tdf[tdf['date'] == date]
            close_price = float(date_rows['close'].iloc[0]) if not date_rows.empty else np.nan

            # y_true_reg = ret_1d of next trading day (log return)
            y_true = np.nan
            d_idx = date_to_idx.get(date)
            if d_idx is not None and d_idx + 1 < len(date_list):
                next_date = date_list[d_idx + 1]
                next_rows = tdf[tdf['date'] == next_date]
                if not next_rows.empty and 'ret_1d' in next_rows.columns:
                    val = next_rows['ret_1d'].iloc[0]
                    if not pd.isna(val):
                        y_true = float(val)

            records.append({
                'date':       date_str,
                'ticker':     ticker,
                'y_pred_reg': y_pred,
                'y_true_reg': y_true,
                'close':      close_price,
            })
            day_preds += 1

        log.info(f"  {date_str}: {day_preds} predictions generated")

    pred_df = pd.DataFrame(records)

    # Keep only tradable dates where y_true_reg exists for at least one ticker.
    # This avoids simulating the most recent prediction day before next-day returns exist.
    if not pred_df.empty and 'y_true_reg' in pred_df.columns:
        by_date_has_truth = pred_df.groupby('date')['y_true_reg'].apply(lambda s: s.notna().any())
        invalid_dates = by_date_has_truth[~by_date_has_truth].index.tolist()
        if invalid_dates:
            log.info(
                "Dropping %d non-tradable date(s) with missing y_true_reg: %s",
                len(invalid_dates), ", ".join(invalid_dates)
            )
            pred_df = pred_df[pred_df['date'].isin(by_date_has_truth[by_date_has_truth].index)].copy()

    log.info(
        f"Total: {len(pred_df)} predictions | "
        f"{pred_df['date'].nunique()} days | "
        f"{pred_df['ticker'].nunique()} tickers"
    )
    return pred_df


def merge_2026_predictions(existing_df: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve previously generated 2026 predictions when an incremental refresh
    produces only future non-tradable dates.
    """
    frames = []

    if existing_df is not None and not existing_df.empty:
        existing = existing_df.copy()
        existing['date'] = pd.to_datetime(existing['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        frames.append(existing)

    if new_df is not None and not new_df.empty:
        incoming = new_df.copy()
        incoming['date'] = pd.to_datetime(incoming['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        frames.append(incoming)

    if not frames:
        return pd.DataFrame(columns=['date', 'ticker', 'y_pred_reg', 'y_true_reg', 'close'])

    merged = pd.concat(frames, ignore_index=True)
    merged = (
        merged
        .sort_values(['date', 'ticker'])
        .drop_duplicates(subset=['date', 'ticker'], keep='last')
        .reset_index(drop=True)
    )
    return merged


# =============================================================================
# Step 5: Combine 2025 + 2026 predictions and run full simulation
# =============================================================================

def combine_and_simulate(pred_2026_df: pd.DataFrame, sim_end: str):
    """
    Merge 2026 predictions with existing 2025 predictions and run
    the full simulation from SIM_START_2025 through sim_end.

    Returns (results_df, combined_pred_df).
    """
    log.info("=" * 60)
    log.info("STEP 5: Combining 2025 + 2026 predictions and simulating")
    log.info("=" * 60)

    # Load 2025 predictions
    pred_2025_df = pd.read_parquet(PRED_2025_PATH)
    pred_2025_df['date'] = pd.to_datetime(pred_2025_df['date']).dt.strftime('%Y-%m-%d')

    # Ensure 2026 dates are strings too
    pred_2026_df = pred_2026_df.copy()
    pred_2026_df['date'] = pd.to_datetime(pred_2026_df['date']).dt.strftime('%Y-%m-%d')

    # Combine, deduplicate (2026 takes precedence over any overlap)
    combined = pd.concat([pred_2025_df, pred_2026_df], ignore_index=True)
    combined = (
        combined
        .sort_values(['date', 'ticker'])
        .drop_duplicates(subset=['date', 'ticker'], keep='last')
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PRED_ALL_PATH, index=False)
    log.info(
        f"Combined: {len(combined)} rows | {combined['date'].nunique()} dates | "
        f"{combined['ticker'].nunique()} tickers"
    )
    log.info(f"  Date range: {combined['date'].min()} -> {combined['date'].max()}")
    log.info(f"  Saved to: {PRED_ALL_PATH}")

    # Run full simulation (single runner instance → correct vol-scaling state)
    log.info(f"Running simulation {SIM_START_2025} -> {sim_end} ...")
    runner = PaperTradingRunner(
        predictions_path=str(PRED_ALL_PATH),
        output_dir=str(OUTPUT_DIR),
    )
    results_df, summary = runner.run_backtest_simulation(
        start_date=SIM_START_2025,
        end_date=sim_end,
    )

    log.info(
        f"Simulation complete: {summary['num_days']} days | "
        f"Sharpe(scaled)={summary['sharpe_scaled']:.2f} | "
        f"MaxDD={summary['max_dd']*100:.1f}%"
    )
    return results_df, combined


# =============================================================================
# Step 6: Sync Qdrant
# =============================================================================

def _load_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL)


def _ensure_collection(client, name: str, size: int, distance_str: str = "Cosine"):
    from qdrant_client.models import Distance, VectorParams
    dist_map = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}
    try:
        client.get_collection(name)
        log.info("Collection '%s' already exists.", name)
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=dist_map[distance_str]),
        )
        log.info("Created collection '%s'.", name)


def _clear_collection(client, name: str):
    from qdrant_client.models import Filter
    client.delete(collection_name=name, points_selector=Filter(must=[]))
    log.info("Cleared '%s'.", name)


def _safe_float(v, default: float = 0.0) -> float:
    """Convert to float and replace NaN/Inf with default (Qdrant rejects non-finite values)."""
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def sync_qdrant(results_df: pd.DataFrame, pred_df: pd.DataFrame):
    """Clear Qdrant collections and re-upload all simulation data."""
    from qdrant_client.models import PointStruct

    log.info("=" * 60)
    log.info("STEP 6: Syncing Qdrant")
    log.info("=" * 60)

    client = _load_qdrant_client()

    _ensure_collection(client, COLLECTION_RECOMMENDATIONS, size=10)
    _ensure_collection(client, COLLECTION_TRADING_RESULTS, size=5)
    _ensure_collection(client, COLLECTION_PERFORMANCE,     size=8)
    _ensure_collection(client, COLLECTION_STRATEGIES,      size=8)

    _clear_collection(client, COLLECTION_RECOMMENDATIONS)
    _clear_collection(client, COLLECTION_TRADING_RESULTS)
    _clear_collection(client, COLLECTION_PERFORMANCE)
    _clear_collection(client, COLLECTION_STRATEGIES)

    dates = sorted(str(d)[:10] for d in results_df['date'].unique())
    total_reco   = 0
    total_result = 0
    total_strategy = 0

    for date_idx, date_str in enumerate(dates):
        date_int = int(date_str.replace('-', ''))
        prev_date = dates[date_idx - 1] if date_idx > 0 else None

        # --- Recommendations ---
        reco_records = build_recommendations_for_date(
            date_str,
            pred_df,
            k=K,
            long_exposure=LONG_EXPOSURE,
            short_exposure=SHORT_EXPOSURE,
        )
        if reco_records:
            reco_points = []
            for idx, rec in enumerate(reco_records):
                point_id = date_int * 10_000 + idx
                vector = [
                    _safe_float(rec['weight']),
                    _safe_float(rec['prediction']),
                    _safe_float(rec['actual_return']),
                    _safe_float(rec['close_price'], 1.0) / 1000.0,
                    1.0 if rec['position'] == 'LONG' else -1.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ]
                reco_points.append(PointStruct(id=point_id, vector=vector, payload=rec))

            client.upsert(collection_name=COLLECTION_RECOMMENDATIONS, points=reco_points)
            total_reco += len(reco_points)

        # --- Daily operation strategy ---
        strategy_payload = build_trade_strategy_payload(
            date_str,
            pred_df,
            previous_date=prev_date,
            k=K,
            long_exposure=LONG_EXPOSURE,
            short_exposure=SHORT_EXPOSURE,
        )
        if strategy_payload:
            strategy_payload['saved_at'] = datetime.now().isoformat()
            client.upsert(
                collection_name=COLLECTION_STRATEGIES,
                points=[PointStruct(
                    id=date_int * 10_000 + 8_888,
                    vector=strategy_payload_to_vector(strategy_payload),
                    payload=strategy_payload,
                )],
            )
            total_strategy += 1

        # --- Trading result ---
        day_row = results_df[results_df['date'] == date_str]
        if not day_row.empty:
            r = day_row.iloc[0].to_dict()
            result_vector = [
                _safe_float(r.get('pnl_net',    0.0)),
                _safe_float(r.get('pnl_scaled', 0.0)),
                _safe_float(r.get('turnover',   0.0)),
                _safe_float(r.get('vol_scale',  1.0)),
                1.0 if r.get('ks_triggered') else 0.0,
            ]
            result_payload = {
                k: (bool(v)             if isinstance(v, (bool, np.bool_))     else
                    _safe_float(v)      if isinstance(v, (float, np.floating)) else
                    int(v)              if isinstance(v, (int,   np.integer))  else v)
                for k, v in r.items()
            }
            result_payload['historical_date'] = date_str

            client.upsert(
                collection_name=COLLECTION_TRADING_RESULTS,
                points=[PointStruct(
                    id=date_int * 10_000 + 9_999,
                    vector=result_vector,
                    payload=result_payload,
                )],
            )
            total_result += 1

    log.info(f"Uploaded {total_reco} recommendation points.")
    log.info(f"Uploaded {total_result} trading result points.")
    log.info(f"Uploaded {total_strategy} strategy points.")

    # --- Performance snapshot ---
    summary_path = OUTPUT_DIR / "phase4_paper_trading_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    perf_vector = [
        _safe_float(summary.get('sharpe_scaled',       0)),
        _safe_float(summary.get('annual_return_scaled', 0)),
        _safe_float(summary.get('vol_scaled',           0)),
        _safe_float(summary.get('max_dd',               0)),
        _safe_float(summary.get('long_sharpe',          0)),
        _safe_float(summary.get('short_sharpe',         0)),
        _safe_float(summary.get('avg_turnover',         0)),
        _safe_float(summary.get('ks_pct',               0)) / 100.0,
    ]
    client.upsert(
        collection_name=COLLECTION_PERFORMANCE,
        points=[PointStruct(
            id=int(datetime.now().timestamp()),
            vector=perf_vector,
            payload={
                **summary,
                'calendar_date':    datetime.now().strftime('%Y-%m-%d'),
                'saved_at':         datetime.now().isoformat(),
                'deployment_ready': (
                    summary.get('sharpe_scaled', 0) > 1.0
                    and summary.get('max_dd', 0) > -0.10
                ),
            },
        )],
    )
    log.info("Uploaded performance snapshot.")


# =============================================================================
# Step 7: Update progress file
# =============================================================================

def update_progress(results_df: pd.DataFrame):
    """Write paper_trading_progress.json."""
    dates     = sorted(str(d)[:10] for d in results_df['date'].unique())
    last_date = dates[-1]

    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
    else:
        progress = {
            'start_calendar_date':  datetime.now().strftime('%Y-%m-%d'),
            'start_historical_date': SIM_START_2025,
        }

    progress['start_historical_date'] = SIM_START_2025
    progress['last_historical_date']  = last_date
    progress['last_calendar_date']    = datetime.now().strftime('%Y-%m-%d')
    progress['days_completed']        = len(dates)

    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

    log.info(f"Progress updated: {len(dates)} days | last: {last_date}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate 2026 live predictions and run combined paper trading'
    )
    parser.add_argument(
        '--skip-download', action='store_true',
        help='Skip all of steps 1-4 and use existing predictions_2026.parquet',
    )
    parser.add_argument(
        '--force-download', action='store_true',
        help='Force re-download of OHLCV data even if cache exists',
    )
    parser.add_argument(
        '--skip-qdrant', action='store_true',
        help='Skip Qdrant sync (only regenerate local files)',
    )
    parser.add_argument(
        '--sim-start', default=SIM_START_2026,
        help=f'First 2026 simulation date (default: {SIM_START_2026})',
    )
    parser.add_argument(
        '--sim-end', default=DEFAULT_SIM_END,
        help="Last 2026 simulation date YYYY-MM-DD (default: auto = yesterday)",
    )
    args = parser.parse_args()
    sim_end = resolve_sim_end(args.sim_end)

    log.info("=" * 60)
    log.info("2026 PAPER TRADING — LIVE DATA")
    log.info(f"  2026 sim range : {args.sim_start} -> {sim_end}")
    log.info(f"  Full range     : {SIM_START_2025} -> {sim_end}")
    log.info(f"  YF download    : {YF_START} -> {YF_END}")
    log.info(f"  Device         : {DEVICE}")
    log.info("=" * 60)

    # ── Steps 1-4: Generate 2026 predictions ─────────────────────────────────
    if args.skip_download and PRED_2026_PATH.exists():
        log.info("Loading existing 2026 predictions from %s", PRED_2026_PATH)
        pred_2026_df = pd.read_parquet(PRED_2026_PATH)
        pred_2026_df['date'] = pd.to_datetime(pred_2026_df['date']).dt.strftime('%Y-%m-%d')
        if 'y_true_reg' in pred_2026_df.columns:
            by_date_has_truth = pred_2026_df.groupby('date')['y_true_reg'].apply(lambda s: s.notna().any())
            pred_2026_df = pred_2026_df[pred_2026_df['date'].isin(by_date_has_truth[by_date_has_truth].index)].copy()
    else:
        # Ticker universe from 2025 predictions
        log.info("Loading ticker universe from phase1_predictions.parquet")
        pred_2025_df = pd.read_parquet(PRED_2025_PATH)
        tickers = sorted(pred_2025_df['ticker'].unique().tolist())
        log.info(f"Ticker universe: {len(tickers)} stocks")

        # Step 1: Download OHLCV (auto-cached; use --force-download to refresh)
        ohlcv_df = download_ohlcv(tickers, required_end_date=sim_end, force=args.force_download)

        # Step 2: Compute features
        features_df = compute_features(ohlcv_df)

        # Step 3: Load model
        model = load_model()

        # Step 4: Generate predictions
        new_pred_2026_df = generate_predictions(features_df, model, args.sim_start, sim_end)

        existing_pred_2026_df = None
        if PRED_2026_PATH.exists():
            try:
                existing_pred_2026_df = pd.read_parquet(PRED_2026_PATH)
                log.info(
                    "Loaded existing 2026 predictions for merge: %d rows | %d dates",
                    len(existing_pred_2026_df),
                    existing_pred_2026_df['date'].nunique() if 'date' in existing_pred_2026_df.columns else 0,
                )
            except Exception as e:
                log.warning("Could not read existing 2026 predictions for merge: %s", e)

        pred_2026_df = merge_2026_predictions(existing_pred_2026_df, new_pred_2026_df)
        if not pred_2026_df.empty:
            pred_2026_df['date'] = pd.to_datetime(pred_2026_df['date']).dt.strftime('%Y-%m-%d')

        # Save 2026 predictions
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pred_2026_df.to_parquet(PRED_2026_PATH, index=False)
        if pred_2026_df.empty:
            log.warning("No tradable 2026 predictions available after merge; saved empty file.")
        else:
            log.info(
                "Saved 2026 predictions: %s (%d rows | %d dates | %s -> %s)",
                PRED_2026_PATH,
                len(pred_2026_df),
                pred_2026_df['date'].nunique(),
                pred_2026_df['date'].min(),
                pred_2026_df['date'].max(),
            )

    # ── Step 5: Combine + simulate ────────────────────────────────────────────
    results_df, combined_pred_df = combine_and_simulate(pred_2026_df, sim_end)

    # ── Step 6: Qdrant sync ───────────────────────────────────────────────────
    if not args.skip_qdrant:
        sync_qdrant(results_df, combined_pred_df)
    else:
        log.info("Skipping Qdrant sync (--skip-qdrant).")

    # ── Step 7: Update progress ───────────────────────────────────────────────
    update_progress(results_df)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("DONE — 2026 paper trading complete")
    log.info(f"  2026 predictions : {PRED_2026_PATH}")
    log.info(f"  Combined preds   : {PRED_ALL_PATH}")
    log.info(f"  Daily results    : {OUTPUT_DIR / 'phase4_paper_trading_daily.parquet'}")
    log.info(f"  Summary          : {OUTPUT_DIR / 'phase4_paper_trading_summary.json'}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
