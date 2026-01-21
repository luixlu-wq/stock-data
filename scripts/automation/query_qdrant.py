"""
Query Qdrant Database

View and search stock recommendations, trading results, and performance metrics
stored in Qdrant vector database.

Usage:
    # View latest recommendations
    python scripts/automation/query_qdrant.py --type recommendations --limit 10

    # View trading results
    python scripts/automation/query_qdrant.py --type results --limit 5

    # View performance metrics
    python scripts/automation/query_qdrant.py --type performance

    # Search similar stocks to AAPL
    python scripts/automation/query_qdrant.py --search AAPL

    # View recommendations for specific date
    python scripts/automation/query_qdrant.py --date 2025-04-01
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Qdrant Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_RECOMMENDATIONS = "stock_recommendations"
COLLECTION_TRADING_RESULTS = "trading_results"
COLLECTION_PERFORMANCE = "performance_metrics"


class QdrantQuery:
    """Query Qdrant database for paper trading data."""

    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT):
        self.client = QdrantClient(host=host, port=port)

    def view_recommendations(self, limit: int = 10, date: str = None):
        """View stock recommendations."""
        print("=" * 80)
        print("STOCK RECOMMENDATIONS")
        print("=" * 80)

        # Build filter if date specified
        scroll_filter = None
        if date:
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="date",
                        match=MatchValue(value=date)
                    )
                ]
            )

        # Scroll through recommendations
        results = self.client.scroll(
            collection_name=COLLECTION_RECOMMENDATIONS,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        if not results[0]:
            print("No recommendations found.")
            return

        # Organize by date
        by_date = {}
        for point in results[0]:
            payload = point.payload
            date_key = payload.get('date', 'Unknown')
            if date_key not in by_date:
                by_date[date_key] = {'longs': [], 'shorts': []}

            if payload.get('position') == 'LONG':
                by_date[date_key]['longs'].append(payload)
            else:
                by_date[date_key]['shorts'].append(payload)

        # Display
        for date_key in sorted(by_date.keys(), reverse=True):
            data = by_date[date_key]
            print(f"\n📅 Date: {date_key}")
            print("-" * 80)

            # Longs
            if data['longs']:
                print(f"\n📈 LONG ({len(data['longs'])} stocks):")
                print(f"{'Ticker':<8} {'Weight':>8} {'Prediction':>12} {'Actual':>12} {'Price':>10}")
                print("-" * 60)
                for stock in sorted(data['longs'], key=lambda x: x.get('weight', 0), reverse=True)[:10]:
                    ticker = stock.get('ticker', 'N/A')
                    weight = stock.get('weight', 0)
                    pred = stock.get('prediction', 0)
                    actual = stock.get('actual_return')
                    price = stock.get('close_price', 0)
                    actual_str = f"{actual:>11.2%}" if actual is not None else "N/A".rjust(12)
                    print(f"{ticker:<8} {weight:>7.2%} {pred:>11.2%} {actual_str} ${price:>9.2f}")

            # Shorts
            if data['shorts']:
                print(f"\n📉 SHORT ({len(data['shorts'])} stocks):")
                print(f"{'Ticker':<8} {'Weight':>8} {'Prediction':>12} {'Actual':>12} {'Price':>10}")
                print("-" * 60)
                for stock in sorted(data['shorts'], key=lambda x: x.get('weight', 0)):[:10]:
                    ticker = stock.get('ticker', 'N/A')
                    weight = stock.get('weight', 0)
                    pred = stock.get('prediction', 0)
                    actual = stock.get('actual_return')
                    price = stock.get('close_price', 0)
                    actual_str = f"{actual:>11.2%}" if actual is not None else "N/A".rjust(12)
                    print(f"{ticker:<8} {weight:>7.2%} {pred:>11.2%} {actual_str} ${price:>9.2f}")

        print("\n" + "=" * 80)

    def view_results(self, limit: int = 5):
        """View daily trading results."""
        print("=" * 80)
        print("DAILY TRADING RESULTS")
        print("=" * 80)

        results = self.client.scroll(
            collection_name=COLLECTION_TRADING_RESULTS,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        if not results[0]:
            print("No trading results found.")
            return

        print(f"\n{'Date':<12} {'PnL Net':>10} {'PnL Scaled':>12} {'Turnover':>10} {'Vol Scale':>10} {'KS':>5}")
        print("-" * 80)

        for point in sorted(results[0], key=lambda x: x.payload.get('historical_date', ''), reverse=True):
            payload = point.payload
            hist_date = payload.get('historical_date', 'N/A')
            pnl_net = payload.get('pnl_net', 0)
            pnl_scaled = payload.get('pnl_scaled', 0)
            turnover = payload.get('turnover', 0)
            vol_scale = payload.get('vol_scale', 0)
            ks = '⚠️' if payload.get('ks_triggered', False) else '✅'

            print(f"{hist_date:<12} {pnl_net:>9.2%} {pnl_scaled:>11.2%} {turnover:>9.1%} {vol_scale:>9.2f} {ks:>5}")

        print("\n" + "=" * 80)

    def view_performance(self):
        """View performance metrics."""
        print("=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)

        results = self.client.scroll(
            collection_name=COLLECTION_PERFORMANCE,
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if not results[0]:
            print("No performance metrics found.")
            return

        latest = results[0][0].payload

        print(f"\n📊 Latest Performance (as of {latest.get('calendar_date', 'N/A')})")
        print("-" * 80)
        print(f"Trading Days:         {latest.get('num_days', 0)}")
        print(f"\nVol-Targeted Performance:")
        print(f"  Sharpe Ratio:       {latest.get('sharpe_scaled', 0):.2f}")
        print(f"  Annual Return:      {latest.get('annual_return_scaled', 0):.2%}")
        print(f"  Volatility:         {latest.get('vol_scaled', 0):.2%}")
        print(f"  Max Drawdown:       {latest.get('max_dd', 0):.2%}")
        print(f"\nAttribution:")
        print(f"  Long Sharpe:        {latest.get('long_sharpe', 0):.2f}")
        print(f"  Short Sharpe:       {latest.get('short_sharpe', 0):.2f}")
        print(f"\nExecution:")
        print(f"  Avg Turnover:       {latest.get('avg_turnover', 0):.2%}")
        print(f"  Avg Cost:           {latest.get('avg_cost_bps', 0):.2f} bps")
        print(f"\nRisk Management:")
        print(f"  Kill Switch Events: {latest.get('ks_events', 0)} ({latest.get('ks_pct', 0):.1f}%)")

        # Deployment status
        deployment_ready = latest.get('deployment_ready', False)
        status = "✅ GREEN LIGHT" if deployment_ready else "⚠️  Review Needed"
        print(f"\nDeployment Status:    {status}")

        print("\n" + "=" * 80)

    def search_similar_stock(self, ticker: str):
        """Find stocks with similar characteristics to given ticker."""
        print("=" * 80)
        print(f"STOCKS SIMILAR TO {ticker}")
        print("=" * 80)

        # First, find the target stock
        target_filter = Filter(
            must=[
                FieldCondition(
                    key="ticker",
                    match=MatchValue(value=ticker)
                )
            ]
        )

        target_results = self.client.scroll(
            collection_name=COLLECTION_RECOMMENDATIONS,
            scroll_filter=target_filter,
            limit=1,
            with_payload=True,
            with_vectors=True
        )

        if not target_results[0]:
            print(f"Stock {ticker} not found in database.")
            return

        target_vector = target_results[0][0].vector
        target_payload = target_results[0][0].payload

        print(f"\n🎯 Target Stock: {ticker}")
        print(f"   Position: {target_payload.get('position', 'N/A')}")
        print(f"   Weight: {target_payload.get('weight', 0):.2%}")
        print(f"   Prediction: {target_payload.get('prediction', 0):.2%}")
        print(f"   Price: ${target_payload.get('close_price', 0):.2f}")

        # Search for similar stocks
        similar = self.client.search(
            collection_name=COLLECTION_RECOMMENDATIONS,
            query_vector=target_vector,
            limit=11  # +1 because first result will be the stock itself
        )

        print(f"\n📊 Similar Stocks:")
        print(f"{'Ticker':<8} {'Position':<6} {'Similarity':>10} {'Weight':>8} {'Prediction':>12}")
        print("-" * 70)

        for i, result in enumerate(similar[1:11], 1):  # Skip first (itself)
            payload = result.payload
            ticker_similar = payload.get('ticker', 'N/A')
            position = payload.get('position', 'N/A')
            similarity = result.score
            weight = payload.get('weight', 0)
            prediction = payload.get('prediction', 0)

            print(f"{ticker_similar:<8} {position:<6} {similarity:>9.3f} {weight:>7.2%} {prediction:>11.2%}")

        print("\n" + "=" * 80)

    def stats(self):
        """Show database statistics."""
        print("=" * 80)
        print("QDRANT DATABASE STATISTICS")
        print("=" * 80)

        for collection_name in [COLLECTION_RECOMMENDATIONS, COLLECTION_TRADING_RESULTS, COLLECTION_PERFORMANCE]:
            try:
                info = self.client.get_collection(collection_name)
                print(f"\n📦 {collection_name}:")
                print(f"   Points: {info.points_count}")
                print(f"   Vectors: {info.vectors_count}")
            except Exception as e:
                print(f"\n📦 {collection_name}: Not found or error ({e})")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Query Qdrant Paper Trading Database')
    parser.add_argument('--type', choices=['recommendations', 'results', 'performance', 'stats'],
                       default='recommendations',
                       help='Type of data to view')
    parser.add_argument('--limit', type=int, default=10,
                       help='Number of records to show')
    parser.add_argument('--date', type=str, default=None,
                       help='Filter by date (YYYY-MM-DD)')
    parser.add_argument('--search', type=str, default=None,
                       help='Search for similar stocks to given ticker')

    args = parser.parse_args()

    query = QdrantQuery()

    try:
        if args.search:
            query.search_similar_stock(args.search)
        elif args.type == 'recommendations':
            query.view_recommendations(limit=args.limit, date=args.date)
        elif args.type == 'results':
            query.view_results(limit=args.limit)
        elif args.type == 'performance':
            query.view_performance()
        elif args.type == 'stats':
            query.stats()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Qdrant server is running!")
        print("Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)


if __name__ == "__main__":
    main()
