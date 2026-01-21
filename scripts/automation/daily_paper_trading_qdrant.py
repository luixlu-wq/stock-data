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
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.paper_trading.phase4_paper_trading_runner import PaperTradingRunner

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


class QdrantPaperTrading:
    """
    Automated paper trading with Qdrant storage.
    """

    def __init__(self, qdrant_host: str = QDRANT_HOST, qdrant_port: int = QDRANT_PORT):
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.predictions_path = "data/processed/phase1_predictions.parquet"
        self.output_dir = "data/processed/phase4"

        # Initialize collections
        self.setup_qdrant_collections()

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

    def get_next_historical_date(self) -> str:
        """Get next historical date based on progress."""
        progress_file = Path(self.output_dir) / "paper_trading_progress.json"

        if not progress_file.exists():
            return "2025-04-01"  # First date

        with open(progress_file, 'r') as f:
            progress = json.load(f)

        last_date = progress.get('last_historical_date', '2025-04-01')

        # Increment to next trading day
        from datetime import timedelta
        last_dt = pd.to_datetime(last_date)
        next_dt = last_dt + timedelta(days=1)

        return next_dt.strftime('%Y-%m-%d')

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

    def run_automated_daily(self):
        """Run complete automated daily workflow."""
        logger.info("="*60)
        logger.info("AUTOMATED DAILY PAPER TRADING")
        logger.info("="*60)
        logger.info(f"Calendar Date: {datetime.now().strftime('%Y-%m-%d')}")

        try:
            # Step 1: Get next historical date
            historical_date = self.get_next_historical_date()
            logger.info(f"Historical Date (simulation): {historical_date}")

            # Step 2: Run paper trading
            result = self.run_daily_paper_trading(historical_date)

            # Step 3: Get stock recommendations
            recommendations = self.get_stock_recommendations(historical_date)

            # Step 4: Save to Qdrant
            self.save_recommendations_to_qdrant(recommendations, historical_date)
            self.save_trading_result_to_qdrant(result, historical_date)
            self.save_performance_metrics_to_qdrant()

            # Step 5: Update progress
            self.update_progress(historical_date)

            # Step 6: Print summary
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

        progress['last_calendar_date'] = datetime.now().strftime('%Y-%m-%d')
        progress['last_historical_date'] = historical_date
        progress['days_completed'] = progress.get('days_completed', 0) + 1

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

        logger.info(f"Progress updated: {progress['days_completed']} days completed")

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


def main():
    """Main entry point for automation."""
    logger.info("Starting automated daily paper trading...")

    try:
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
