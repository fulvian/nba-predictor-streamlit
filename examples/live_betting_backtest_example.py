#!/usr/bin/env python3
"""
Example: Live Betting Arbitrage Backtest

Demonstrates how to:
1. Load historical odds data (CSV)
2. Run anomaly detection
3. Generate betting opportunities
4. Simulate trades and calculate P&L
5. Generate performance report

Usage:
    python examples/live_betting_backtest_example.py \
        --odds-file path/to/odds_history.csv \
        --initial-bankroll 10000 \
        --stake-pct 1.0

Example odds CSV format:
    timestamp,sport,competition,event_id,bookmaker,market_type,outcome,odds,backing_odds,laying_odds,back_volume,lay_volume
    2025-12-29T14:00:00Z,football,Serie C,1.12345,betfair,WIN,home,2.50,2.50,2.48,1000,1500
    2025-12-29T14:00:05Z,football,Serie C,1.12345,betfair,WIN,draw,3.00,3.00,2.98,800,1200
    ...
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from live_betting import AnomalyDetector, ArbitrageEngine
from live_betting.backtest import BacktestEngine
from live_betting.anomaly_detector import OddsSnapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_sample_odds_data() -> list:
    """
    Create synthetic odds data for demo purposes.
    In production, load from historical Betfair data.
    """
    from datetime import datetime, timedelta
    import random

    snapshots = []
    base_time = datetime(2025, 12, 29, 14, 0, 0)
    
    # Simulate 3 events with live odds evolution
    events = [
        {"event_id": "1.190001", "sport": "football", "competition": "Serie C"},
        {"event_id": "1.190002", "sport": "tennis", "competition": "ITF Women"},
        {"event_id": "1.190003", "sport": "basketball", "competition": "Basketball League"},
    ]
    
    for event_num, event_info in enumerate(events):
        # Pre-match baseline (recorded as baseline)
        baseline_time = base_time + timedelta(hours=event_num)
        
        for minute in range(-30, 100, 2):  # -30 min (pre-match) to 100 min
            time = baseline_time + timedelta(minutes=minute)
            
            # Simulate odds evolution
            # Start at baseline, add random walk with some mean reversion
            if minute < 0:
                # Pre-match: stable
                odds_home = 2.50 + random.gauss(0, 0.05)
                odds_draw = 3.00 + random.gauss(0, 0.05)
                odds_away = 2.80 + random.gauss(0, 0.05)
            else:
                # Live: assume some random movements
                # Occasionally introduce larger moves (mimicking "anomalies")
                if minute % 20 == 0:
                    shock = random.gauss(0, 0.3)  # Occasional larger move
                else:
                    shock = random.gauss(0, 0.05)
                
                odds_home = 2.50 + shock
                odds_draw = 3.00 + shock * 0.5
                odds_away = 2.80 - shock * 0.3
            
            # Generate snapshots for each outcome
            for outcome_name, base_odds in [("home", odds_home), ("draw", odds_draw), ("away", odds_away)]:
                # Simulate bid-ask spread (more during anomalies)
                spread = 0.02 if abs(shock if minute > 0 else 0) < 0.1 else 0.05
                
                snapshot = OddsSnapshot(
                    timestamp=time,
                    sport=event_info["sport"],
                    competition=event_info["competition"],
                    event_id=event_info["event_id"],
                    bookmaker="betfair",
                    market_type="WIN",
                    outcome=outcome_name,
                    odds=max(1.01, base_odds),  # Ensure valid odds
                    backing_odds=max(1.01, base_odds - spread/2),
                    laying_odds=max(1.01, base_odds + spread/2),
                    back_volume=random.uniform(100, 5000),
                    lay_volume=random.uniform(100, 5000),
                    implied_prob=1.0 / max(1.01, base_odds),
                )
                snapshots.append(snapshot)
    
    return snapshots


def main():
    parser = argparse.ArgumentParser(
        description="Run live betting arbitrage backtest"
    )
    parser.add_argument(
        "--odds-file",
        type=str,
        default=None,
        help="Path to CSV file with odds history (if not provided, uses synthetic data)",
    )
    parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=10000.0,
        help="Starting capital for backtest",
    )
    parser.add_argument(
        "--stake-pct",
        type=float,
        default=1.0,
        help="Percentage of bankroll to stake per trade",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./backtest_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        default=True,
        help="Use synthetic data for demo (default: True)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Live Betting Arbitrage Backtest")
    logger.info("=" * 70)
    logger.info(f"Initial Bankroll: ${args.initial_bankroll:,.2f}")
    logger.info(f"Stake per trade: {args.stake_pct}%")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    
    # Initialize backtest engine
    bt = BacktestEngine(
        initial_bankroll=args.initial_bankroll,
        stake_pct_per_trade=args.stake_pct,
    )
    
    # Load odds data
    logger.info("\n[STEP 1] Loading odds data...")
    if args.odds_file and Path(args.odds_file).exists():
        logger.info(f"Loading from file: {args.odds_file}")
        success = bt.load_odds_file(args.odds_file)
        if not success:
            logger.error("Failed to load odds file; using synthetic data instead")
            snapshots = create_sample_odds_data()
            bt.load_odds_list(snapshots)
    else:
        logger.info("No odds file provided; using synthetic data for demo")
        snapshots = create_sample_odds_data()
        bt.load_odds_list(snapshots)
    
    logger.info(f"Total snapshots: {len(bt.odds_snapshots)}")
    
    # Run backtest
    logger.info("\n[STEP 2] Running backtest...")
    bt.run()
    
    # Generate report
    logger.info("\n[STEP 3] Analyzing results...")
    report = bt.generate_report()
    
    # Display metrics
    if "metrics" in report:
        metrics = report["metrics"]
        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 70)
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"  Winning: {metrics['winning_trades']}")
        logger.info(f"  Losing: {metrics['losing_trades']}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"\nProfitability:")
        logger.info(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
        logger.info(f"  Return %: {metrics['total_pnl_pct']:.2f}%")
        logger.info(f"  Avg Win: ${metrics['avg_win']:,.2f}")
        logger.info(f"  Avg Loss: ${metrics['avg_loss']:,.2f}")
        logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"\nRisk Metrics:")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"\nAnomaly Detection:")
        logger.info(f"  Anomalies Detected: {metrics['anomalies_detected']}")
        logger.info(f"  Opportunities Generated: {metrics['opportunities_generated']}")
        logger.info(f"  Trades per Anomaly: {metrics['trades_per_anomaly']:.2f}")
    else:
        logger.warning(f"Report: {report}")
    
    # Export results
    logger.info("\n[STEP 4] Exporting results...")
    
    # Save report JSON
    report_file = output_dir / "backtest_report.json"
    with open(report_file, "w") as f:
        # Convert datetime objects to strings for JSON serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        json.dump(report, f, indent=2, default=json_serializer)
    logger.info(f"Report saved to: {report_file}")
    
    # Export trades CSV
    trades_file = output_dir / "trades.csv"
    bt.export_trades_csv(str(trades_file))
    
    # Equity curve data
    equity_data = bt.plot_equity_curve()
    if equity_data:
        equity_file = output_dir / "equity_curve.json"
        with open(equity_file, "w") as f:
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            json.dump(equity_data, f, indent=2, default=json_serializer)
        logger.info(f"Equity curve saved to: {equity_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Backtest complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
