"""
Daily Paper Trading with Qdrant Storage

Automated daily paper trading that:
1. Runs paper trading for today
2. Saves all trading info to Qdrant vector DB
3. Stores stock recommendations with full specifications
4. Tracks performance metrics over time

This script is designed to be run by a cron job/scheduler.

Usage:
    python scripts/automation/daily_paper_trading_qdrant.py
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Ensure working directory is project root (critical for Task Scheduler)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from scripts.paper_trading.phase4_paper_trading_runner import PaperTradingRunner

# Docker/Qdrant Configuration
QDRANT_CONTAINER_NAME = "qdrant-paper-trading"
QDRANT_IMAGE = "qdrant/qdrant"
DOCKER_DESKTOP_PATH = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Backup Configuration
BACKUP_DIR = Path("data/backups/qdrant")

# Setup logging
log_dir = Path("logs/automation")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"daily_automation_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Qdrant Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_RECOMMENDATIONS = "stock_recommendations"
COLLECTION_TRADING_RESULTS = "trading_results"
COLLECTION_PERFORMANCE = "performance_metrics"
PHASE1_PREDICTIONS = Path("data/processed/phase1_predictions.parquet")
COMBINED_PREDICTIONS = Path("data/processed/phase4/predictions_combined.parquet")


class QdrantPaperTrading:
    """
    Automated paper trading with Qdrant storage.
    """

    def __init__(self, qdrant_host: str = QDRANT_HOST, qdrant_port: int = QDRANT_PORT):
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.predictions_path = str(self.resolve_predictions_path())
        self.output_dir = "data/processed/phase4"

        # Initialize collections
        self.setup_qdrant_collections()

    @staticmethod
    def resolve_predictions_path() -> Path:
        """Use combined predictions when available; otherwise fallback to phase1."""
        if COMBINED_PREDICTIONS.exists():
            return COMBINED_PREDICTIONS
        return PHASE1_PREDICTIONS

    @staticmethod
    def load_available_dates(predictions_path: str) -> tuple[list[str], list[str]]:
        """Load (all_dates, tradable_dates) as YYYY-MM-DD strings."""
        try:
            df = pd.read_parquet(predictions_path, columns=['date', 'y_true_reg'])
        except Exception:
            df = pd.read_parquet(predictions_path, columns=['date'])

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        df = df.dropna(subset=['date'])

        all_dates = sorted(df['date'].unique().tolist())
        if not all_dates:
            raise ValueError(f"No valid dates found in predictions: {predictions_path}")

        if 'y_true_reg' in df.columns:
            by_date_has_truth = df.groupby('date')['y_true_reg'].apply(lambda s: s.notna().any())
            tradable_dates = sorted(by_date_has_truth[by_date_has_truth].index.tolist())
        else:
            tradable_dates = all_dates

        if not tradable_dates:
            raise ValueError(f"No tradable dates (with y_true_reg) found in predictions: {predictions_path}")

        return all_dates, tradable_dates

    @staticmethod
    def build_refresh_command(max_date: str) -> str:
        next_start = (pd.to_datetime(max_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        return (
            f"python scripts/paper_trading/run_2026_paper_trading.py "
            f"--sim-start {next_start} --sim-end auto"
        )

    def get_predictions_status(self) -> Dict[str, str]:
        """
        Return current predictions coverage status.

        Keys:
            min_tradable, max_tradable, max_raw, is_stale, stale_cutoff
        """
        all_dates, tradable_dates = self.load_available_dates(self.predictions_path)
        min_tradable = tradable_dates[0]
        max_tradable = tradable_dates[-1]
        max_raw = all_dates[-1]
        stale_cutoff = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        is_stale = max_tradable < stale_cutoff
        return {
            'min_tradable': min_tradable,
            'max_tradable': max_tradable,
            'max_raw': max_raw,
            'is_stale': 'true' if is_stale else 'false',
            'stale_cutoff': stale_cutoff,
        }

    def refresh_predictions_if_stale(self) -> bool:
        """
        Refresh predictions using run_2026_paper_trading.py when tradable data is stale.

        Returns True if a refresh command was executed (successfully), else False.
        """
        status = self.get_predictions_status()
        if status['is_stale'] != 'true':
            return False

        max_tradable = status['max_tradable']
        next_start = (pd.to_datetime(max_tradable) + timedelta(days=1)).strftime('%Y-%m-%d')
        cmd = [
            sys.executable,
            "scripts/paper_trading/run_2026_paper_trading.py",
            "--sim-start",
            next_start,
            "--sim-end",
            "auto",
        ]

        logger.warning(
            "Predictions are stale (max tradable %s, expected >= %s). Auto-refreshing...",
            max_tradable,
            status['stale_cutoff'],
        )
        logger.info("Refresh command: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Prediction refresh failed (exit=%s)", result.returncode)
            if result.stdout:
                logger.error("Refresh stdout:\n%s", result.stdout[-4000:])
            if result.stderr:
                logger.error("Refresh stderr:\n%s", result.stderr[-4000:])
            return False

        logger.info("Prediction refresh completed successfully.")
        if result.stdout:
            logger.info("Refresh stdout tail:\n%s", result.stdout[-1000:])
        return True

    def log_no_data_diagnosis(self):
        """
        Emit a concise no-action diagnosis so operators can quickly tell why
        the 4:15 PM scheduler run did not place trades.
        """
        progress_file = Path(self.output_dir) / "paper_trading_progress.json"
        last_processed = None
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                last_processed = progress.get('last_historical_date')
            except Exception:
                last_processed = None

        status = self.get_predictions_status()
        stale_text = (
            f"stale predictions (max tradable {status['max_tradable']} < {status['stale_cutoff']})"
            if status['is_stale'] == 'true'
            else "predictions not stale"
        )
        logger.info(
            "Scheduler run completed with no trade: no tradable historical date after last_processed=%s (%s).",
            last_processed or "N/A",
            stale_text,
        )

    def setup_qdrant_collections(self):
        """Create Qdrant collections if they don't exist."""
        logger.info("Setting up Qdrant collections...")

        # Collection 1: Stock Recommendations (with embeddings for similarity search)
        # Vector dimension: We'll use a simple embedding based on stock features
        try:
            self.qdrant_client.get_collection(COLLECTION_RECOMMENDATIONS)
            logger.info(f"Collection '{COLLECTION_RECOMMENDATIONS}' already exists")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_RECOMMENDATIONS,
                vectors_config=VectorParams(size=10, distance=Distance.COSINE),
            )
            logger.info(f"Created collection '{COLLECTION_RECOMMENDATIONS}'")

        # Collection 2: Daily Trading Results
        try:
            self.qdrant_client.get_collection(COLLECTION_TRADING_RESULTS)
            logger.info(f"Collection '{COLLECTION_TRADING_RESULTS}' already exists")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_TRADING_RESULTS,
                vectors_config=VectorParams(size=5, distance=Distance.COSINE),
            )
            logger.info(f"Created collection '{COLLECTION_TRADING_RESULTS}'")

        # Collection 3: Performance Metrics
        try:
            self.qdrant_client.get_collection(COLLECTION_PERFORMANCE)
            logger.info(f"Collection '{COLLECTION_PERFORMANCE}' already exists")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_PERFORMANCE,
                vectors_config=VectorParams(size=8, distance=Distance.COSINE),
            )
            logger.info(f"Created collection '{COLLECTION_PERFORMANCE}'")

    def get_next_historical_date(self) -> str | None:
        """Get next historical date based on progress, skipping weekends/holidays."""
        progress_file = Path(self.output_dir) / "paper_trading_progress.json"

        all_dates, available_dates = self.load_available_dates(self.predictions_path)
        min_date = available_dates[0]
        max_date = available_dates[-1]
        max_raw_date = all_dates[-1]

        if not progress_file.exists():
            return min_date

        with open(progress_file, 'r') as f:
            progress = json.load(f)

        last_date = progress.get('last_historical_date')
        if not last_date:
            return min_date

        # Find the next date after last_date (strip timezone for comparison)
        last_dt = pd.to_datetime(last_date, errors='coerce')
        if pd.isna(last_dt):
            logger.warning("Invalid last_historical_date '%s' in progress file. Resetting to %s.", last_date, min_date)
            return min_date
        last_dt = last_dt.tz_localize(None)
        max_dt = pd.to_datetime(max_date)
        if last_dt > max_dt:
            logger.warning(
                "Progress file date %s is ahead of available tradable data (max %s). Clamping.",
                last_date, max_date
            )
            last_dt = max_dt

        for d in available_dates:
            d_naive = pd.to_datetime(d)
            if d_naive > last_dt:
                return d_naive.strftime('%Y-%m-%d')

        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        logger.info(
            "No new historical dates available. Last processed: %s | Tradable range: %s to %s",
            last_date, min_date, max_date
        )
        if max_raw_date > max_date:
            logger.info(
                "Newest raw prediction date is %s but not tradable yet (y_true_reg missing).",
                max_raw_date
            )
        if max_date < yesterday:
            logger.warning("Predictions look stale. Refresh with:")
            logger.warning("  %s", self.build_refresh_command(max_date))
        return None

    def run_daily_paper_trading(self, historical_date: str) -> Dict:
        """Run paper trading for the specified date."""
        logger.info(f"Running paper trading for {historical_date}")

        runner = PaperTradingRunner(
            predictions_path=self.predictions_path,
            output_dir=self.output_dir
        )

        result = runner.run_single_day(historical_date)

        if not result:
            raise ValueError(f"Paper trading failed for {historical_date}")

        logger.info(f"Paper trading complete: PnL={result['pnl_net']:.4f}, Sharpe cumulative ongoing")

        return result

    def get_stock_recommendations(self, historical_date: str) -> Dict[str, List[Dict]]:
        """Get detailed stock recommendations for the date."""
        logger.info(f"Loading stock recommendations for {historical_date}")

        # Load predictions
        df = pd.read_parquet(self.predictions_path)
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        day = df[df['date'] == historical_date].copy()

        if day.empty:
            raise ValueError(f"No data for date {historical_date}")

        # Sort by prediction
        day_sorted = day.sort_values('y_pred_reg', ascending=False)

        # Configuration
        K = 38
        LONG_EXPOSURE = 0.65
        SHORT_EXPOSURE = 0.35

        # Long positions
        longs = day_sorted.head(K).copy()
        long_weight = LONG_EXPOSURE / len(longs)

        long_recommendations = []
        for _, row in longs.iterrows():
            long_recommendations.append({
                'ticker': row['ticker'],
                'position': 'LONG',
                'weight': float(long_weight),
                'prediction': float(row['y_pred_reg']),
                'actual_return': float(row['y_true_reg']) if not pd.isna(row['y_true_reg']) else None,
                'close_price': float(row['close']),
                'date': historical_date
            })

        # Short positions (filtered)
        shorts_candidates = day_sorted.tail(K)
        shorts = shorts_candidates[shorts_candidates['y_pred_reg'] < 0].copy()

        short_recommendations = []
        if len(shorts) > 0:
            short_weight = -SHORT_EXPOSURE / len(shorts)

            for _, row in shorts.iterrows():
                short_recommendations.append({
                    'ticker': row['ticker'],
                    'position': 'SHORT',
                    'weight': float(short_weight),
                    'prediction': float(row['y_pred_reg']),
                    'actual_return': float(row['y_true_reg']) if not pd.isna(row['y_true_reg']) else None,
                    'close_price': float(row['close']),
                    'date': historical_date
                })

        return {
            'longs': long_recommendations,
            'shorts': short_recommendations
        }

    def create_stock_vector(self, stock: Dict) -> List[float]:
        """
        Create a simple vector representation for stock recommendation.
        This enables similarity searches in Qdrant.
        """
        return [
            float(stock['weight']),
            float(stock['prediction']),
            float(stock['actual_return']) if stock['actual_return'] is not None else 0.0,
            float(stock['close_price']) / 1000.0,  # Normalized price
            1.0 if stock['position'] == 'LONG' else -1.0,
            # Add padding to reach dimension 10
            0.0, 0.0, 0.0, 0.0, 0.0
        ]

    def save_recommendations_to_qdrant(self, recommendations: Dict[str, List[Dict]], historical_date: str):
        """Save stock recommendations to Qdrant."""
        logger.info("Saving recommendations to Qdrant...")

        points = []
        point_id = int(datetime.now().timestamp() * 1000)  # Unique ID based on timestamp

        # Save longs
        for stock in recommendations['longs']:
            point_id += 1
            vector = self.create_stock_vector(stock)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        **stock,
                        'calendar_date': datetime.now().strftime('%Y-%m-%d'),
                        'saved_at': datetime.now().isoformat()
                    }
                )
            )

        # Save shorts
        for stock in recommendations['shorts']:
            point_id += 1
            vector = self.create_stock_vector(stock)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        **stock,
                        'calendar_date': datetime.now().strftime('%Y-%m-%d'),
                        'saved_at': datetime.now().isoformat()
                    }
                )
            )

        # Batch upload to Qdrant
        self.qdrant_client.upsert(
            collection_name=COLLECTION_RECOMMENDATIONS,
            points=points
        )

        logger.info(f"Saved {len(points)} stock recommendations to Qdrant")

    def create_result_vector(self, result: Dict) -> List[float]:
        """Create vector for trading result."""
        return [
            float(result['pnl_net']),
            float(result['pnl_scaled']),
            float(result['turnover']),
            float(result['vol_scale']),
            1.0 if result['ks_triggered'] else 0.0
        ]

    def save_trading_result_to_qdrant(self, result: Dict, historical_date: str):
        """Save daily trading result to Qdrant."""
        logger.info("Saving trading result to Qdrant...")

        point_id = int(datetime.now().timestamp() * 1000)
        vector = self.create_result_vector(result)

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                **result,
                'historical_date': historical_date,
                'calendar_date': datetime.now().strftime('%Y-%m-%d'),
                'saved_at': datetime.now().isoformat()
            }
        )

        self.qdrant_client.upsert(
            collection_name=COLLECTION_TRADING_RESULTS,
            points=[point]
        )

        logger.info(f"Saved trading result to Qdrant")

    def save_performance_metrics_to_qdrant(self):
        """Save cumulative performance metrics to Qdrant."""
        logger.info("Saving performance metrics to Qdrant...")

        summary_file = Path(self.output_dir) / "phase4_paper_trading_summary.json"

        if not summary_file.exists():
            logger.warning("Summary file not found, skipping performance metrics")
            return

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        # Create vector from key metrics
        vector = [
            float(summary.get('sharpe_scaled', 0)),
            float(summary.get('annual_return_scaled', 0)),
            float(summary.get('vol_scaled', 0)),
            float(summary.get('max_dd', 0)),
            float(summary.get('long_sharpe', 0)),
            float(summary.get('short_sharpe', 0)),
            float(summary.get('avg_turnover', 0)),
            float(summary.get('ks_pct', 0)) / 100.0
        ]

        point_id = int(datetime.now().timestamp() * 1000)

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                **summary,
                'calendar_date': datetime.now().strftime('%Y-%m-%d'),
                'saved_at': datetime.now().isoformat(),
                'deployment_ready': summary.get('sharpe_scaled', 0) > 1.0 and summary.get('max_dd', 0) > -0.10
            }
        )

        self.qdrant_client.upsert(
            collection_name=COLLECTION_PERFORMANCE,
            points=[point]
        )

        logger.info(f"Saved performance metrics to Qdrant")

    def save_daily_results_and_summary(self, result: Dict):
        """Append daily result to parquet and regenerate summary metrics."""
        daily_file = Path(self.output_dir) / "phase4_paper_trading_daily.parquet"

        # Load existing or create new DataFrame
        if daily_file.exists():
            existing_df = pd.read_parquet(daily_file)
            results_df = pd.concat([existing_df, pd.DataFrame([result])], ignore_index=True)
        else:
            results_df = pd.DataFrame([result])

        # Save daily parquet
        results_df.to_parquet(daily_file)
        logger.info(f"Saved {len(results_df)} daily results to {daily_file}")

        # Calculate and save summary metrics
        pnl_net = results_df['pnl_net'].values
        pnl_scaled = results_df['pnl_scaled'].values

        sharpe_raw = (np.mean(pnl_net) / np.std(pnl_net)) * np.sqrt(252) if np.std(pnl_net) > 0 else 0
        sharpe_scaled = (np.mean(pnl_scaled) / np.std(pnl_scaled)) * np.sqrt(252) if np.std(pnl_scaled) > 0 else 0

        cum_returns = np.cumsum(pnl_net)
        running_max = np.maximum.accumulate(cum_returns)
        max_dd = float(np.min(cum_returns - running_max))

        long_pnl = results_df['long_pnl'].values
        short_pnl = results_df['short_pnl'].values
        long_sharpe = (np.mean(long_pnl) / np.std(long_pnl)) * np.sqrt(252) if np.std(long_pnl) > 0 else 0
        short_sharpe = (np.mean(short_pnl) / np.std(short_pnl)) * np.sqrt(252) if np.std(short_pnl) > 0 else 0

        ks_events = int(results_df['ks_triggered'].sum())

        summary = {
            'num_days': len(results_df),
            'sharpe_raw': float(sharpe_raw),
            'annual_return_raw': float(np.mean(pnl_net) * 252),
            'vol_raw': float(np.std(pnl_net) * np.sqrt(252)),
            'sharpe_scaled': float(sharpe_scaled),
            'annual_return_scaled': float(np.mean(pnl_scaled) * 252),
            'vol_scaled': float(np.std(pnl_scaled) * np.sqrt(252)),
            'max_dd': max_dd,
            'avg_turnover': float(results_df['turnover'].mean()),
            'avg_cost_bps': float(results_df['cost'].mean() * 10000),
            'long_sharpe': float(long_sharpe),
            'short_sharpe': float(short_sharpe),
            'ks_events': ks_events,
            'ks_pct': (ks_events / len(results_df)) * 100
        }

        summary_file = Path(self.output_dir) / "phase4_paper_trading_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary: Sharpe(scaled)={sharpe_scaled:.2f}, Days={len(results_df)}")

    def run_automated_daily(self):
        """Run complete automated daily workflow."""
        logger.info("="*60)
        logger.info("AUTOMATED DAILY PAPER TRADING")
        logger.info("="*60)
        logger.info(f"Calendar Date: {datetime.now().strftime('%Y-%m-%d')}")
        logger.info("Predictions source: %s", self.predictions_path)

        try:
            # Step 1: Get next historical date
            historical_date = self.get_next_historical_date()
            if historical_date is None:
                refreshed = self.refresh_predictions_if_stale()
                if refreshed:
                    historical_date = self.get_next_historical_date()

            if historical_date is None:
                logger.info("No action taken today because there is no new historical date to process.")
                self.log_no_data_diagnosis()
                logger.info("="*60)
                logger.info("DAILY AUTOMATION COMPLETE (NO NEW DATA)")
                logger.info("="*60)
                return
            logger.info(f"Historical Date (simulation): {historical_date}")

            # Step 2: Run paper trading
            result = self.run_daily_paper_trading(historical_date)

            # Step 3: Save daily results and summary to local files
            self.save_daily_results_and_summary(result)

            # Step 4: Get stock recommendations
            recommendations = self.get_stock_recommendations(historical_date)

            # Step 5: Save to Qdrant
            self.save_recommendations_to_qdrant(recommendations, historical_date)
            self.save_trading_result_to_qdrant(result, historical_date)
            self.save_performance_metrics_to_qdrant()

            # Step 6: Update progress
            self.update_progress(historical_date)

            # Step 7: Incremental backup
            self.backup_qdrant_data(historical_date)

            # Step 8: Print summary
            self.print_summary(result, recommendations)

            logger.info("="*60)
            logger.info("DAILY AUTOMATION COMPLETE")
            logger.info("="*60)

        except Exception as e:
            logger.error(f"Automation failed: {e}", exc_info=True)
            raise

    def update_progress(self, historical_date: str):
        """Update progress tracker."""
        progress_file = Path(self.output_dir) / "paper_trading_progress.json"

        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {
                'start_calendar_date': datetime.now().strftime('%Y-%m-%d'),
                'start_historical_date': '2025-04-01',
                'days_completed': 0
            }

        previous_date = progress.get('last_historical_date')
        progress['last_calendar_date'] = datetime.now().strftime('%Y-%m-%d')
        progress['last_historical_date'] = historical_date
        if previous_date != historical_date:
            progress['days_completed'] = progress.get('days_completed', 0) + 1

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

        logger.info(f"Progress updated: {progress['days_completed']} days completed")

    def backup_qdrant_data(self, historical_date: str):
        """Incremental backup of Qdrant data to local JSON files after each trading day."""
        today = datetime.now().strftime('%Y%m%d')
        backup_dir = BACKUP_DIR / today
        backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Backing up Qdrant data to {backup_dir}...")

        for collection_name in [COLLECTION_RECOMMENDATIONS, COLLECTION_TRADING_RESULTS, COLLECTION_PERFORMANCE]:
            try:
                info = self.qdrant_client.get_collection(collection_name)
                total_points = info.points_count

                # Scroll through all points
                all_points = []
                offset = None
                while True:
                    results, next_offset = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        limit=100,
                        offset=offset,
                        with_vectors=True
                    )
                    for point in results:
                        all_points.append({
                            'id': point.id,
                            'vector': point.vector,
                            'payload': point.payload
                        })
                    if next_offset is None:
                        break
                    offset = next_offset

                backup_file = backup_dir / f"{collection_name}.json"
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'collection': collection_name,
                        'backup_date': datetime.now().isoformat(),
                        'historical_date': historical_date,
                        'total_points': len(all_points),
                        'points': all_points
                    }, f, indent=2, default=str)

                logger.info(f"  Backed up {len(all_points)} points from '{collection_name}'")

            except Exception as e:
                logger.error(f"  Failed to backup '{collection_name}': {e}")

        # Write a manifest file with backup summary
        manifest = {
            'backup_date': datetime.now().isoformat(),
            'historical_date': historical_date,
            'calendar_date': today,
            'collections': {}
        }
        for f in backup_dir.glob("*.json"):
            if f.name != "manifest.json":
                with open(f, 'r') as fh:
                    data = json.load(fh)
                    manifest['collections'][f.stem] = data['total_points']

        with open(backup_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Backup complete: {backup_dir / 'manifest.json'}")

    def print_summary(self, result: Dict, recommendations: Dict):
        """Print summary of today's trading."""
        print("\n" + "="*60)
        print("TODAY'S SUMMARY")
        print("="*60)
        print(f"PnL Net: {result['pnl_net']:.4f} ({result['pnl_net']*100:.2f}%)")
        print(f"PnL Scaled: {result['pnl_scaled']:.4f} ({result['pnl_scaled']*100:.2f}%)")
        print(f"Turnover: {result['turnover']:.2%}")
        print(f"Vol Scale: {result['vol_scale']:.2f}")
        print(f"\nPortfolio:")
        print(f"  Longs: {len(recommendations['longs'])} stocks")
        print(f"  Shorts: {len(recommendations['shorts'])} stocks")
        print(f"\nKill Switches: {'TRIGGERED' if result['ks_triggered'] else 'OK'}")
        print("="*60)


def is_docker_desktop_running() -> bool:
    """Check if Docker Desktop process is running on Windows."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Docker Desktop.exe", "/NH"],
            capture_output=True, text=True, timeout=10
        )
        return "Docker Desktop.exe" in result.stdout
    except Exception:
        return False


def ensure_docker_running() -> bool:
    """Check Docker Desktop process and Docker daemon, start if needed."""
    # Check if Docker daemon is already responsive
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            logger.info("Docker Desktop is running, daemon is ready")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Docker daemon not responding - check Docker Desktop process
    if is_docker_desktop_running():
        logger.info("Docker Desktop process is running but daemon not ready, waiting...")
    else:
        logger.warning("Docker Desktop is not running, starting it...")
        if os.path.exists(DOCKER_DESKTOP_PATH):
            subprocess.Popen([DOCKER_DESKTOP_PATH], shell=False)
            logger.info(f"Launched Docker Desktop from {DOCKER_DESKTOP_PATH}")
        else:
            logger.error(f"Docker Desktop not found at {DOCKER_DESKTOP_PATH}")
            return False

    # Wait for Docker daemon to become ready (up to 90 seconds)
    for i in range(18):
        time.sleep(5)
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info(f"Docker daemon ready after {(i+1)*5} seconds")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        logger.info(f"Waiting for Docker daemon... ({(i+1)*5}s)")

    logger.error("Docker daemon failed to start within 90 seconds")
    return False


def ensure_qdrant_running() -> bool:
    """Check if Qdrant container is running, start it if not."""
    # First check if Qdrant is reachable
    try:
        resp = requests.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}", timeout=3)
        if resp.status_code == 200:
            logger.info("Qdrant is already running")
            return True
    except requests.ConnectionError:
        pass

    logger.warning("Qdrant is not reachable, checking Docker container...")

    # Check if container exists (running or stopped)
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={QDRANT_CONTAINER_NAME}",
         "--format", "{{.Status}}"],
        capture_output=True, text=True, timeout=10
    )
    container_status = result.stdout.strip()

    if container_status:
        if container_status.startswith("Up"):
            logger.info(f"Container '{QDRANT_CONTAINER_NAME}' is up, waiting for port...")
        else:
            # Container exists but is stopped - restart it
            logger.info(f"Starting stopped container '{QDRANT_CONTAINER_NAME}'...")
            subprocess.run(
                ["docker", "start", QDRANT_CONTAINER_NAME],
                capture_output=True, text=True, timeout=30
            )
    else:
        # Also check for any qdrant container with a different name
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "ancestor=qdrant/qdrant",
             "--format", "{{.Names}} {{.Status}}"],
            capture_output=True, text=True, timeout=10
        )
        existing = result.stdout.strip()

        if existing:
            # Found a qdrant container with different name
            name = existing.split()[0]
            status = " ".join(existing.split()[1:])
            logger.info(f"Found existing Qdrant container '{name}' ({status})")
            if not status.startswith("Up"):
                logger.info(f"Starting container '{name}'...")
                subprocess.run(
                    ["docker", "start", name],
                    capture_output=True, text=True, timeout=30
                )
        else:
            # No qdrant container exists - create one
            logger.info(f"Creating new Qdrant container '{QDRANT_CONTAINER_NAME}'...")
            subprocess.run(
                ["docker", "run", "-d",
                 "-p", "6333:6333", "-p", "6334:6334",
                 "--name", QDRANT_CONTAINER_NAME,
                 QDRANT_IMAGE],
                capture_output=True, text=True, timeout=60
            )

    # Wait for Qdrant to become ready (up to 30 seconds)
    for i in range(10):
        time.sleep(3)
        try:
            resp = requests.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}", timeout=3)
            if resp.status_code == 200:
                logger.info(f"Qdrant ready after {(i+1)*3} seconds")
                return True
        except requests.ConnectionError:
            pass
        logger.info(f"Waiting for Qdrant... ({(i+1)*3}s)")

    logger.error("Qdrant failed to start within 30 seconds")
    return False


def main():
    """Main entry point for automation."""
    logger.info("Starting automated daily paper trading...")

    try:
        # Step 0: Ensure Docker and Qdrant are running
        if not ensure_docker_running():
            logger.error("Cannot proceed without Docker")
            sys.exit(1)

        if not ensure_qdrant_running():
            logger.error("Cannot proceed without Qdrant")
            sys.exit(1)

        # Create automation instance
        automation = QdrantPaperTrading()

        # Run daily workflow
        automation.run_automated_daily()

        logger.info("Automation completed successfully!")

    except Exception as e:
        logger.error(f"Automation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
