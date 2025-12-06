#!/usr/bin/env python3
"""
Bankroll Debug Script - Simple and Direct Analysis
"""

import duckdb
import json
from pathlib import Path

DB_PATH = "data/nba_betting.duckdb"
INITIAL_DEPOSIT = 84.75


def debug_bankroll():
    print("=== BANKROLL DEBUG ===")

    try:
        conn = duckdb.connect(DB_PATH)

        # 1. Get raw bet data
        print("\n1. RAW BET DATA:")
        bets = conn.execute("SELECT * FROM bets ORDER BY created_at ASC").fetchall()
        print(f"Total bets: {len(bets)}")

        # Print column names to understand structure
        columns = [desc[0] for desc in conn.execute("DESCRIBE bets").fetchall()]
        print(f"Columns: {columns}")

        # 2. Analyze each bet properly
        print("\n2. DETAILED BET ANALYSIS:")
        print(
            f"{'Index':<6} | {'Status':<8} | {'Amount':<8} | {'Odds':<5} | {'P/L':<8} | {'Created':<20}"
        )
        print("-" * 70)

        total_pl = 0.0
        pending_amount = 0.0
        won_count = 0
        lost_count = 0
        pending_count = 0

        for i, bet in enumerate(bets):
            # Create dict for easier access
            bet_dict = dict(zip(columns, bet))

            status = bet_dict.get("status", "UNKNOWN")
            amount = float(bet_dict.get("amount", 0))
            odds = float(bet_dict.get("odds", 0))
            profit_loss = bet_dict.get("profit_loss")
            created_at = str(bet_dict.get("created_at", ""))[:19]

            # Count by status
            if status == "WON":
                won_count += 1
            elif status == "LOST":
                lost_count += 1
            elif status == "PENDING":
                pending_count += 1
                pending_amount += amount

            # Calculate P/L if not stored
            if profit_loss is not None:
                try:
                    profit_loss = float(profit_loss)
                    total_pl += profit_loss
                except:
                    profit_loss = 0.0
            else:
                # Calculate P/L from status
                if status == "WON":
                    profit_loss = amount * (odds - 1)  # Net profit
                    total_pl += profit_loss
                elif status == "LOST":
                    profit_loss = -amount
                    total_pl += profit_loss
                else:
                    profit_loss = 0.0

            print(
                f"{i:<6} | {status:<8} | €{amount:<7.2f} | {odds:<5.2f} | €{profit_loss:<7.2f} | {created_at:<20}"
            )

        # 3. Summary
        print("\n3. SUMMARY:")
        print(f"  Won bets: {won_count}")
        print(f"  Lost bets: {lost_count}")
        print(f"  Pending bets: {pending_count}")
        print(f"  Pending amount: €{pending_amount:.2f}")
        print(f"  Total P/L: €{total_pl:.2f}")

        # 4. Bankroll calculations
        print("\n4. BANKROLL CALCULATIONS:")
        method1_total = INITIAL_DEPOSIT + total_pl
        method1_free = method1_total - pending_amount

        print(f"  Initial deposit: €{INITIAL_DEPOSIT:.2f}")
        print(f"  Total P/L: €{total_pl:.2f}")
        print(f"  Total equity: €{method1_total:.2f}")
        print(f"  Pending stakes: €{pending_amount:.2f}")
        print(f"  Free bankroll: €{method1_free:.2f}")

        # 5. Check transactions table
        print("\n5. TRANSACTIONS TABLE:")
        try:
            tx_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            tx_sum = conn.execute(
                "SELECT SUM(amount) FROM transactions WHERE user_id = 'test_user_001'"
            ).fetchone()[0]
            print(f"  Transaction count: {tx_count}")
            print(f"  Transaction sum: €{tx_sum:.2f if tx_sum else 0:.2f}")
        except Exception as e:
            print(f"  Error reading transactions: {e}")

        # 6. Check bankroll file
        print("\n6. BANKROLL FILE:")
        bankroll_path = Path("data/bankroll.json")
        if bankroll_path.exists():
            with open(bankroll_path, "r") as f:
                bankroll_data = json.load(f)
            current_file_bankroll = bankroll_data.get("current_bankroll", 0)
            print(f"  File bankroll: €{current_file_bankroll:.2f}")
        else:
            print("  Bankroll file not found")

        # 7. Expected vs Actual
        print("\n7. ANALYSIS:")
        print(f"  Expected total equity (~€97.00): €{method1_total:.2f}")
        print(f"  Expected free bankroll: €{method1_free:.2f}")

        if "current_file_bankroll" in locals():
            diff = method1_free - current_file_bankroll
            print(f"  Difference from file: €{diff:.2f}")

        conn.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_bankroll()
