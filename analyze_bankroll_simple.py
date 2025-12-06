#!/usr/bin/env python3
"""
Simple Bankroll Analysis Script
Analyzes the betting database to understand bankroll calculation issues
"""

import duckdb
import json
from pathlib import Path

DB_PATH = "data/nba_betting.duckdb"
INITIAL_DEPOSIT = 84.75


def analyze_database():
    """Analyze the betting database directly"""
    print("=== BANKROLL ANALYSIS ===")

    try:
        conn = duckdb.connect(DB_PATH)

        # 1. Check table structure
        print("\n1. TABLE STRUCTURE:")
        try:
            bets_schema = conn.execute("DESCRIBE bets").fetchall()
            print("Bets table columns:")
            for col in bets_schema:
                print(f"  - {col[0]} ({col[1]})")
        except Exception as e:
            print(f"Error getting schema: {e}")

        # Check if transactions table exists
        try:
            tx_schema = conn.execute("DESCRIBE transactions").fetchall()
            print("Transactions table exists:")
            for col in tx_schema:
                print(f"  - {col[0]} ({col[1]})")
        except Exception as e:
            print("Transactions table does not exist or error:", e)

        # 2. Get all bets with their current status
        print("\n2. ALL BETS ANALYSIS:")
        bets_query = """
            SELECT 
                bet_id,
                amount,
                odds,
                status,
                result,
                profit_loss,
                created_at,
                settled_at,
                home_score,
                away_score
            FROM bets 
            ORDER BY created_at ASC
        """

        bets = conn.execute(bets_query).fetchall()
        print(f"Total bets found: {len(bets)}")

        # 3. Calculate totals by status
        print("\n3. BETS BY STATUS:")
        status_summary = {}
        for bet in bets:
            status = bet[2]  # status column
            if status not in status_summary:
                status_summary[status] = {
                    "count": 0,
                    "total_amount": 0.0,
                    "total_profit_loss": 0.0,
                }

            status_summary[status]["count"] += 1
            status_summary[status]["total_amount"] += float(bet[1])  # amount

            if bet[4] is not None:  # profit_loss
                try:
                    status_summary[status]["total_profit_loss"] += float(bet[4])
                except (ValueError, TypeError):
                    # profit_loss might be NULL or invalid
                    pass

        for status, data in status_summary.items():
            print(
                f"  {status}: {data['count']} bets, Total: €{data['total_amount']:.2f}, P/L: €{data['total_profit_loss']:.2f}"
            )

        # 4. Calculate bankroll using different methods
        print("\n4. BANKROLL CALCULATION METHODS:")

        # Method 1: Initial + P/L
        total_profit_loss = 0.0
        for bet in bets:
            profit_loss = bet[8]  # profit_loss column
            if profit_loss is not None:
                try:
                    total_profit_loss += float(profit_loss)
                except (ValueError, TypeError):
                    pass
        method1_bankroll = INITIAL_DEPOSIT + total_profit_loss
        print(f"  Method 1 (Initial + P/L): €{method1_bankroll:.2f}")

        # Method 2: Sum from transactions (if exists)
        try:
            tx_result = conn.execute(
                "SELECT SUM(amount) FROM transactions WHERE user_id = 'test_user_001'"
            ).fetchone()
            if tx_result and tx_result[0]:
                method2_bankroll = float(tx_result[0])
                print(f"  Method 2 (Transactions sum): €{method2_bankroll:.2f}")
            else:
                print("  Method 2: No transactions found")
        except Exception as e:
            print(f"  Method 2: Error - {e}")

        # Method 3: Current free + pending
        pending_total = sum(float(bet[1]) for bet in bets if bet[2] == "PENDING")
        method3_free = method1_bankroll - pending_total
        print(f"  Method 3 (Free bankroll): €{method3_free:.2f}")
        print(f"  Method 3 (Pending): €{pending_total:.2f}")
        print(f"  Method 3 (Total equity): €{method1_bankroll:.2f}")

        # 5. Check bankroll file
        print("\n5. BANKROLL FILE:")
        bankroll_path = Path("data/bankroll.json")
        if bankroll_path.exists():
            with open(bankroll_path, "r") as f:
                bankroll_data = json.load(f)
            print(
                f"  Current bankroll file: €{bankroll_data.get('current_bankroll', 0):.2f}"
            )
            print(f"  File contents: {bankroll_data}")
        else:
            print("  Bankroll file not found")

        # 6. Detailed bet analysis
        print("\n6. DETAILED BET ANALYSIS:")
        print(
            f"{'Date':<20} | {'Status':<8} | {'Amount':<8} | {'Odds':<5} | {'P/L':<8} | {'Running':<8}"
        )
        print("-" * 70)

        running_total = INITIAL_DEPOSIT
        for bet in bets:
            created_at = (
                str(bet[10])[:19] if bet[10] else "Unknown"
            )  # created_at is bet[10]
            status = bet[6]
            amount = float(bet[4])
            odds = float(bet[5]) if bet[5] else 0.0
            profit_loss = float(bet[8]) if bet[8] is not None else 0.0

            # Update running total
            if status != "PENDING":
                running_total += profit_loss

            print(
                f"{created_at:<20} | {status:<8} | €{amount:<7.2f} | {odds:<5.2f} | €{profit_loss:<7.2f} | €{running_total:<7.2f}"
            )

        conn.close()

    except Exception as e:
        print(f"Error analyzing database: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    analyze_database()
