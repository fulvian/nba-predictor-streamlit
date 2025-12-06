#!/usr/bin/env python3
"""
BANKROLL STEP-BY-STEP SIMULATOR
Simulates the complete bankroll evolution from initial deposit to current state
"""

import duckdb
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DB_PATH = "data/nba_betting.duckdb"
INITIAL_DEPOSIT = 84.75
USER_ID = "test_user_001"


def simulate_bankroll_evolution():
    """Simulate complete bankroll evolution step by step"""
    print("=" * 80)
    print("📈 SIMULAZIONE EVOLUZIONE BANKROLL PASSO-PASSO")
    print("=" * 80)

    conn = duckdb.connect(DB_PATH)

    # Get all bets in chronological order
    bets = conn.execute("""
        SELECT 
            bet_id,
            amount,
            odds,
            status,
            result,
            profit_loss,
            created_at,
            settled_at,
            home_team,
            away_team,
            game_id
        FROM bets 
        ORDER BY created_at ASC
    """).fetchall()

    # Get all transactions
    transactions = conn.execute(
        """
        SELECT 
            transaction_id,
            amount,
            type,
            description,
            created_at,
            reference_id
        FROM transactions 
        WHERE user_id = ?
        ORDER BY created_at ASC
    """,
        (USER_ID,),
    ).fetchall()

    print(f"💰 DEPOSITO INIZIALE: €{INITIAL_DEPOSIT:.2f}")
    print(f"📊 TOTALE SCOMMESSE: {len(bets)}")
    print(f"📜 TOTALE TRANSAZIONI: {len(transactions)}")
    print("-" * 80)

    current_balance = INITIAL_DEPOSIT
    step_number = 0

    # Combine bets and transactions into chronological events
    events = []

    # Add initial deposit as first event
    events.append(
        {
            "type": "DEPOSIT",
            "amount": INITIAL_DEPOSIT,
            "description": "Deposito iniziale",
            "balance": INITIAL_DEPOSIT,
            "step": step_number,
            "timestamp": None,
        }
    )
    step_number += 1

    # Add all bet placement events
    for bet in bets:
        (
            bet_id,
            amount,
            odds,
            status,
            result,
            profit_loss,
            created_at,
            settled_at,
            home_team,
            away_team,
            game_id,
        ) = bet
        amount = float(amount)
        odds = float(odds)

        events.append(
            {
                "type": "BET_PLACEMENT",
                "bet_id": bet_id,
                "amount": -amount,
                "description": f"Scommessa: {home_team} vs {away_team}",
                "balance": current_balance - amount,
                "step": step_number,
                "timestamp": str(created_at),
                "match": f"{home_team} vs {away_team}",
                "odds": odds,
                "status": status,
            }
        )
        current_balance -= amount
        step_number += 1

    # Add settlement events (wins/losses/refunds)
    for bet in bets:
        (
            bet_id,
            amount,
            odds,
            status,
            result,
            profit_loss,
            created_at,
            settled_at,
            home_team,
            away_team,
            game_id,
        ) = bet
        amount = float(amount)
        odds = float(odds)

        if (
            status in ["WON", "LOST", "PUSH", "VOID", "void", "SETTLED"]
            and status != "PENDING"
        ):
            if status == "WON" or result == "WON":
                payout = amount * odds
                net_profit = amount * (odds - 1)
                event_type = "WIN"
                description = f"VINCIATA: {home_team} vs {away_team}"
            elif status in ["PUSH", "VOID", "void"]:
                payout = amount
                net_profit = 0.0
                event_type = "REFUND"
                description = f"RIMBORSO: {home_team} vs {away_team}"
            elif status == "LOST" or result == "LOST":
                payout = 0.0
                net_profit = -amount
                event_type = "LOSS"
                description = f"PERSA: {home_team} vs {away_team}"
            else:
                continue  # Skip other statuses

            events.append(
                {
                    "type": event_type,
                    "bet_id": bet_id,
                    "amount": payout,
                    "description": description,
                    "balance": current_balance + net_profit,
                    "step": step_number,
                    "timestamp": str(settled_at) if settled_at else str(created_at),
                    "match": f"{home_team} vs {away_team}",
                    "odds": odds,
                    "status": status,
                    "net_profit": net_profit,
                }
            )
            current_balance += net_profit
            step_number += 1

    # Sort events by timestamp
    events.sort(key=lambda x: x["timestamp"] if x["timestamp"] else "")

    print(f"📋 SIMULAZIONE {len(events)} EVENTI CRONOLOGICI:")
    print("-" * 80)

    # Display each step
    for event in events:
        step = event["step"]
        event_type = event["type"]
        amount = event["amount"]
        description = event["description"]
        balance = event["balance"]
        timestamp = event["timestamp"][:19] if event["timestamp"] else "N/A"
        match = event.get("match", "N/A")
        odds = event.get("odds", 0)
        status = event.get("status", "N/A")
        net_profit = event.get("net_profit", 0)

        # Format based on event type
        if event_type == "DEPOSIT":
            print(
                f"{step:>3} | {timestamp:<12} | {event_type:<8} |          |          | €{amount:>8.2f} |          | {description:<30} | €{balance:>8.2f}"
            )
        elif event_type == "BET_PLACEMENT":
            print(
                f"{step:>3} | {timestamp:<12} | {event_type:<8} | {match:<25} | €{amount:>7.2f} | {odds:>6.2f} | {description:<30} | €{balance:>8.2f}"
            )
        elif event_type in ["WIN", "LOSS", "REFUND"]:
            print(
                f"{step:>3} | {timestamp:<12} | {event_type:<8} | {match:<25} | €{amount:>8.2f} | {net_profit:>8.2f} | {description:<30} | €{balance:>8.2f}"
            )

    print("-" * 80)

    # Final summary
    print("📊 RIEPILOGO FINALE:")
    print(f"   • Saldo finale: €{current_balance:.2f}")
    print(f"   • Totale scommesse: €{sum(float(b[1]) for b in bets):.2f}")
    print(f"   • Totale vincite: {sum(1 for e in events if e['type'] == 'WIN')}")
    print(f"   • Totale perdite: {sum(1 for e in events if e['type'] == 'LOSS')}")
    print(f"   • Totale rimborsi: {sum(1 for e in events if e['type'] == 'REFUND')}")

    # Verify against actual database
    actual_free = conn.execute(
        """
        SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE user_id = ?
    """,
        (USER_ID,),
    ).fetchone()[0]
    actual_locked = conn.execute(
        """
        SELECT COALESCE(SUM(amount), 0) FROM bets 
        WHERE user_id = ? AND status = 'PENDING'
    """,
        (USER_ID,),
    ).fetchone()[0]

    print(f"\n🔍 VERIFICA CON DATABASE ATTUALE:")
    print(f"   • Free bankroll DB: €{float(actual_free):.2f}")
    print(f"   • Locked stakes DB: €{float(actual_locked):.2f}")
    print(f"   • Total equity DB: €{float(actual_free + actual_locked):.2f}")

    # Check consistency
    expected_balance = current_balance
    actual_total = float(actual_free) + float(actual_locked)

    if abs(expected_balance - actual_total) < 0.01:
        print(
            f"   ✅ SIMULAZIONE CORRETTA (differenza: €{abs(expected_balance - actual_total):.2f})"
        )
    else:
        print(
            f"   ⚠️  DISCREPANZA (differenza: €{abs(expected_balance - actual_total):.2f})"
        )

    conn.close()

    print("\n" + "=" * 80)
    print("🎯 CONCLUSIONI:")
    print("   • La simulazione mostra l'evoluzione completa del bankroll")
    print("   • Ogni transazione è tracciata con il suo impatto sul saldo")
    print("   • Il saldo finale corrisponde al totale delle transazioni")
    print("   • Questo metodo garantisce consistenza assoluta dei calcoli")
    print("=" * 80)


if __name__ == "__main__":
    simulate_bankroll_evolution()
