import json
import logging
import sys
from typing import List, Dict, Any

import pandas as pd
import numpy as np

# Adjust path to find src
sys.path.append(".")

from src.nba_predictor.bankroll.engine import TransactionEngine, BetRecord, BetResult
from src.nba_predictor.intelligence.bias_corrector import get_bias_corrector
from src.nba_predictor.analytics.ev_calculator import EVCalculator
from src.nba_predictor.analytics.betting_filters import get_betting_filters

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/nba_bankroll_v3.duckdb"
CSV_PATH = "data/risultati_bet_completi.csv"


def parse_line_from_selection(selection: str) -> float:
    # Format usually "Over 225.5" or "Under 230.0"
    try:
        parts = str(selection).split()
        return float(parts[-1])
    except (IndexError, ValueError):
        return 0.0


class MockBetRecord:
    def __init__(self, row):
        # CSV Cols: Squadra A, Squadra B, Data, Ora, Quota, Stake, Media Punti Stimati, Probabilita Stimata, Edge Value, Confidenza, Punteggio finale, Esito, Tipo Scommessa
        try:
            raw_pred = float(row.get("Media Punti Stimati", 0))
        except:
            raw_pred = 0.0

        try:
            model_prob = float(row.get("Probabilita Stimata", 0)) / 100.0
        except:
            model_prob = 0.0

        self.metadata = {
            "predicted_total": raw_pred,
            "model_probability": model_prob,
            "line": 0.0,
        }

        try:
            quota = float(str(row["Quota"]).replace(",", "."))
        except:
            quota = 2.0

        self.odds = int((quota - 1) * 100) if quota >= 2 else int(-100 / (quota - 1))

        try:
            self.stake = float(str(row["Stake"]).replace(",", "."))
        except:
            self.stake = 0.0

        self.result = BetResult.WON if row["Esito"] == "W" else BetResult.LOST

        payout = self.stake * (quota - 1)
        self.profit_loss = payout if self.result == BetResult.WON else -self.stake

        self.selection = row.get("Tipo Scommessa", "")
        self.bet_type = "OVER" if "Over" in str(self.selection) else "UNDER"

        edge_val = 0.0
        try:
            edge_val = float(str(row.get("Edge Value", 0)).replace(",", "."))
        except:
            pass

        if self.bet_type == "OVER" and raw_pred > 0 and edge_val != 0:
            self.metadata["line"] = raw_pred - edge_val
        elif self.bet_type == "UNDER" and raw_pred > 0 and edge_val != 0:
            self.metadata["line"] = raw_pred + edge_val


def run_simulation():
    print("=" * 60)
    print("BACKTEST SIMULATION: CURRENT LOGIC ON HISTORICAL DATA")
    print("=" * 60)

    bets = []

    # Try DB First
    try:
        engine = TransactionEngine(DB_PATH)
        bets = engine.get_bet_records(limit=500)
    except Exception as e:
        logger.warning(f"DB Fetch failed: {e}")

    if not bets:
        print("DB empty or failed. Trying CSV...")
        try:
            df = pd.read_csv(CSV_PATH)
            # Filter valid rows
            df = df[df["Esito"].isin(["W", "L"])]
            print(f"Loaded {len(df)} bets from CSV.")
            bets = [MockBetRecord(row) for _, row in df.iterrows()]
        except Exception as e:
            logger.error(f"CSV Fetch failed: {e}")
            return

    print(f"Total bets found: {len(bets)}")

    # Initialize components
    corrector = get_bias_corrector(enabled=True)
    filters = get_betting_filters(strict_mode=True)
    ev_calc = EVCalculator(bankroll=100.0, min_edge=0.08, min_model_prob=0.58)

    # Stats
    original_stats = {"placed": 0, "won": 0, "lost": 0, "pnl": 0.0, "invested": 0.0}
    new_stats = {"placed": 0, "won": 0, "lost": 0, "pnl": 0.0, "invested": 0.0}
    blocked_stats = {"count": 0, "pnl_avoided": 0.0}

    for bet in bets:
        # 1. Parse Data
        # Based on debug output: bet.selection contains the JSON metadata!
        # bet.bet_type contains "OVER 225.0" logic?

        metadata = {}
        # Try parsing selection as JSON
        if isinstance(bet.selection, str) and "{" in bet.selection:
            try:
                metadata = json.loads(bet.selection)
            except:
                pass

        # If selection wasn't JSON, maybe metadata is?
        if not metadata and hasattr(bet, "metadata"):
            if isinstance(bet.metadata, str) and "{" in bet.metadata:
                try:
                    metadata = json.loads(bet.metadata)
                except:
                    pass
            elif isinstance(bet.metadata, dict):
                metadata = bet.metadata

        # Extract fields
        market_line = 0.0

        # Try to get line from bet_type "OVER 225.5"
        try:
            parts = str(bet.bet_type).split()  # e.g. ["OVER", "225.0"]
            for p in parts:
                try:
                    val = float(p)
                    if val > 150:  # Reasonable total line
                        market_line = val
                except:
                    pass
        except:
            pass

        if market_line == 0.0 and metadata:
            market_line = float(metadata.get("market_line", 0.0))
            if market_line == 0.0:
                market_line = float(metadata.get("line", 0.0))

        model_prob = 0.0
        if metadata:
            if "over_probability" in metadata:
                # Check SELECTION for direction, as bet_type is often just "Total"
                selection_upper = str(bet.selection).upper()
                is_over = "OVER" in selection_upper

                if is_over:
                    model_prob = float(metadata["over_probability"])
                else:
                    model_prob = float(metadata.get("under_probability", 0.0))
                    if model_prob == 0 and "over_probability" in metadata:
                        model_prob = 1.0 - float(metadata["over_probability"])
            elif "model_probability" in metadata:
                model_prob = float(metadata["model_probability"])
            elif "win_probability" in metadata:
                model_prob = float(metadata["win_probability"])

        # Original Outcome
        is_win = bet.result == BetResult.WON
        is_loss = bet.result == BetResult.LOST
        pnl = bet.profit_loss
        stake = bet.stake

        # DEBUG
        if original_stats["placed"] == 0:
            print(
                f"DEBUG: Bet 1: Line={market_line}, Prob={model_prob}, Type={bet.bet_type}"
            )

        if is_win:
            original_stats["won"] += 1
        elif is_loss:
            original_stats["lost"] += 1

        original_stats["placed"] += 1
        original_stats["pnl"] += float(pnl)
        original_stats["invested"] += float(stake)

        # 2. Apply Refounded Architecture Logic (Simulation)
        should_bet = False
        filter_reason = "PASS"

        # We need:
        # 1. Raw Prediction (bet.metadata["predicted_total"])
        # 2. Market Line
        # 3. Dynamic Bias (simulated or 0 if unknown)

        raw_pred = 0.0
        if metadata:
            raw_pred = float(metadata.get("predicted_total", 0.0))

        if raw_pred > 0 and market_line > 0:
            # A. Dynamic Bias (Mocking conservative momentum for simulation)
            # In real system this comes from DB. Here we assume neutral to test Shrinkage pure impact.
            dynamic_bias = 0.0
            base_pred = raw_pred + dynamic_bias

            # B. Bayesian Shrinkage (The Fail-Safe)
            shrunk_pred, weight, shrink_status = corrector.apply_bayesian_shrinkage(
                base_pred, market_line
            )

            # C. Validator / Edge Check
            # Calculate new edge based on Shrunk Prediction
            if bet.bet_type == "OVER":
                new_edge = shrunk_pred - market_line
            else:
                new_edge = market_line - shrunk_pred

            # Min Edge Reqt (e.g. 1.5 pts post-shrinkage)
            if new_edge >= 1.5:
                # D. Risk Check (Validator)
                # If we had consensus risk in metadata we'd check it.
                # Here we simulate proper filtering.
                if shrink_status == "CRITICAL_SHRINK" or weight < 0.4:
                    should_bet = False
                    filter_reason = (
                        f"SHRINKAGE_KILL (W={weight:.2f}, Status={shrink_status})"
                    )
                else:
                    should_bet = True
            else:
                should_bet = False
                filter_reason = f"NO_EDGE_POST_SHRINK (Edge={new_edge:.1f})"
        else:
            # If data missing, default to NO BERT in simulation to be safe
            should_bet = False
            filter_reason = "MISSING_DATA"

        # 3. Simulate Result
        if should_bet:
            new_stats["placed"] += 1
            new_stats["invested"] += float(stake)
            new_stats["pnl"] += float(pnl)
            if is_win:
                new_stats["won"] += 1
            elif is_loss:
                new_stats["lost"] += 1
        else:
            # Blocked
            blocked_stats["count"] += 1
            blocked_stats["pnl_avoided"] += float(pnl)

    # Report
    print("\n" + "=" * 40)
    print("RESULTS COMPARISON")
    print("=" * 40)

    orig_roi = (
        (original_stats["pnl"] / original_stats["invested"] * 100)
        if original_stats["invested"]
        else 0
    )
    new_roi = (
        (new_stats["pnl"] / new_stats["invested"] * 100) if new_stats["invested"] else 0
    )

    print(f"ORIGINAL:")
    print(f"  Bets: {original_stats['placed']}")
    print(f"  Won:  {original_stats['won']}")
    print(f"  Lost: {original_stats['lost']}")
    print(f"  P&L:  €{original_stats['pnl']:.2f}")
    print(f"  ROI:  {orig_roi:.2f}%")

    print(f"\nSIMULATED (New filters):")
    print(f"  Bets: {new_stats['placed']}")
    print(f"  Won:  {new_stats['won']}")
    print(f"  Lost: {new_stats['lost']}")
    print(f"  P&L:  €{new_stats['pnl']:.2f}")
    print(f"  ROI:  {new_roi:.2f}%")

    pnl_diff = new_stats["pnl"] - original_stats["pnl"]
    print(f"\nIMPACT:")
    print(f"  Blocked Bets: {blocked_stats['count']}")
    print(f"  P&L Diff: €{pnl_diff:+.2f} (Value Added)")
    print(f"  Avoided P&L: €{blocked_stats['pnl_avoided']:.2f}")

    if blocked_stats["pnl_avoided"] < 0:
        print("  ✅ Filters avoided losses!")
    elif blocked_stats["pnl_avoided"] > 0:
        print("  ⚠️ Filters blocked winning bets.")


if __name__ == "__main__":
    run_simulation()
