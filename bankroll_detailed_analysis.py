#!/usr/bin/env python3
"""
DETAILED BANKROLL ANALYSIS - Complete Breakdown
Shows all bets, transactions, and calculations for verification
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


def analyze_all_bets_detailed():
    """Analyze all bets with complete breakdown"""
    print("=" * 80)
    print("📊 ANALISI DETTAGLIATA COMPLETA DEL BANKROLL")
    print("=" * 80)

    conn = duckdb.connect(DB_PATH)

    # 1. Get all bets with full details
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
            home_score,
            away_score,
            home_team,
            away_team,
            game_id
        FROM bets 
        ORDER BY created_at ASC
    """).fetchall()

    print(f"\n📈 TOTALE SCOMMESSE: {len(bets)}")
    print("-" * 80)

    # 2. Detailed breakdown by status
    print(
        f"{'#':<4} | {'Data':<12} | {'Squadra':<15} | {'Tipo':<8} | {'Importo':<8} | {'Quota':<6} | {'Risultato':<8} | {'P/L':<8} | {'Payout':<8} | {'Saldo':<8}"
    )
    print("-" * 120)

    running_balance = INITIAL_DEPOSIT
    total_won = 0
    total_lost = 0
    total_pending = 0
    total_pending_amount = 0.0

    for i, bet in enumerate(bets, 1):
        bet_id = bet[0]
        amount = float(bet[1])
        odds = float(bet[2])
        status = bet[3]
        result = bet[4] or ""
        profit_loss = bet[5] if bet[5] is not None else 0.0
        created_at = str(bet[6])[:19] if bet[6] else ""
        home_team = bet[10] if len(bet) > 10 else ""
        away_team = bet[11] if len(bet) > 11 else ""

        # Calculate payout and net profit
        payout = 0.0
        net_profit = 0.0

        if status == "WON" or result == "WON":
            payout = amount * odds
            net_profit = amount * (odds - 1)
            total_won += 1
        elif status == "LOST" or result == "LOST":
            payout = 0.0
            net_profit = -amount
            total_lost += 1
        elif status == "PENDING":
            total_pending += 1
            total_pending_amount += amount
            payout = 0.0
            net_profit = 0.0
        elif status in ["PUSH", "VOID", "void", "SETTLED"]:
            payout = amount  # Refund
            net_profit = 0.0

        # Update running balance (only for settled bets)
        if status != "PENDING":
            running_balance += net_profit

        # Display bet details
        match_info = (
            f"{home_team} vs {away_team}"
            if home_team and away_team
            else f"Game {bet[12] if len(bet) > 12 else ''}"
        )

        print(
            f"{i:<3} | {created_at:<12} | {match_info:<25} | {status:<8} | €{amount:<7.2f} | {odds:<6.2f} | €{payout:<8.2f} | €{net_profit:<8.2f} | €{running_balance:<8.2f}"
        )

    # 3. Summary statistics
    print("-" * 120)
    print("📊 RIEPILOGO STATISTICHE:")
    print(f"   • Scommesse VINTE: {total_won}")
    print(f"   • Scommesse PERSE: {total_lost}")
    print(f"   • Scommesse IN ATTESA: {total_pending}")
    print(f"   • Importo totale scommesse: €{sum(float(b[1]) for b in bets):.2f}")
    print(f"   • Importo totale PENDING: €{total_pending_amount:.2f}")

    # 4. Financial summary
    total_pl = sum(
        float(b[5]) if b[5] is not None else 0.0 for b in bets if b[3] != "PENDING"
    )
    expected_final_balance = INITIAL_DEPOSIT + total_pl

    print(f"\n💰 RIEPILOGO FINANZIARIO:")
    print(f"   • Deposito iniziale: €{INITIAL_DEPOSIT:.2f}")
    print(f"   • Profit/Loss totale: €{total_pl:.2f}")
    print(f"   • Saldo finale atteso: €{expected_final_balance:.2f}")
    print(f"   • Saldo attuale calcolato: €{running_balance:.2f}")
    print(f"   • Importo bloccato (PENDING): €{total_pending_amount:.2f}")
    print(
        f"   • Disponibile per scommesse: €{running_balance - total_pending_amount:.2f}"
    )

    # 5. Transaction analysis
    print(f"\n📜 ANALISI TRANSACTIONS:")
    try:
        transactions = conn.execute(
            """
            SELECT transaction_id, amount, type, description, created_at, reference_id
            FROM transactions 
            WHERE user_id = ?
            ORDER BY created_at ASC
        """,
            (USER_ID,),
        ).fetchall()

        print(f"   • Totale transazioni: {len(transactions)}")

        tx_summary = {}
        for tx in transactions:
            tx_type = tx[2]
            amount = float(tx[1])
            if tx_type not in tx_summary:
                tx_summary[tx_type] = {"count": 0, "total": 0.0}
            tx_summary[tx_type]["count"] += 1
            tx_summary[tx_type]["total"] += amount

        for tx_type, data in tx_summary.items():
            print(
                f"   • {tx_type}: {data['count']} transazioni, totale €{data['total']:.2f}"
            )

    except Exception as e:
        print(f"   • Errore lettura transazioni: {e}")

    # 6. Current bankroll file
    print(f"\n📁 STATO FILE BANKROLL:")
    bankroll_path = Path("data/bankroll.json")
    if bankroll_path.exists():
        with open(bankroll_path, "r") as f:
            bankroll_data = json.load(f)
        print(
            f"   • Free bankroll nel file: €{bankroll_data.get('current_bankroll', 0):.2f}"
        )
        print(
            f"   • Locked bankroll nel file: €{bankroll_data.get('locked_bankroll', 0):.2f}"
        )
        print(
            f"   • Total equity nel file: €{bankroll_data.get('total_equity', 0):.2f}"
        )
        print(f"   • Last updated: {bankroll_data.get('last_updated', 'N/A')}")
    else:
        print("   • File bankroll.json non trovato")

    conn.close()

    print("\n" + "=" * 80)
    print("🔍 VERIFICA CONSISTENZA:")
    print("   • Controlla che Total Equity (Initial + P/L) corrisponda al saldo finale")
    print("   • Verifica che le transazioni sommano correttamente al free bankroll")
    print("   • Assicura che le scommesse PENDING siano sottratte correttamente")
    print("=" * 80)


if __name__ == "__main__":
    analyze_all_bets_detailed()
