#!/usr/bin/env python3
"""
Test semplificato per verificare solo le nuove funzionalità di gestione scommesse.
"""

import sys
sys.path.append('/Users/fulvioventura/nba-predictor-streamlit/src')

import streamlit as st
import pandas as pd
from datetime import datetime, date
from nba_predictor.utils.betting_database_manager import BettingDatabaseManager

def main():
    """Dashboard semplificato per testare le nuove funzionalità."""
    st.set_page_config(
        page_title="NBA Betting Management - Test",
        page_icon="🏀",
        layout="wide"
    )

    st.title("🏀 NBA Betting Management - Test Nuove Funzionalità")
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("📋 Navigazione")
    page = st.sidebar.selectbox("Seleziona Pagina", [
        "📊 Dashboard Principale",
        "💰 Gestione Scommesse",
        "🎯 Test Funzionalità"
    ])

    if page == "📊 Dashboard Principale":
        render_dashboard_principale()
    elif page == "💰 Gestione Scommesse":
        render_gestione_scommesse()
    elif page == "🎯 Test Funzionalità":
        render_test_funzionalita()

def render_dashboard_principale():
    """Dashboard principale con stato del sistema."""
    st.header("📊 Dashboard Principale")

    try:
        with BettingDatabaseManager() as db_manager:
            # Get bankroll status
            bankroll = db_manager.get_bankroll_status()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("💰 Bankroll Attuale", f"€{bankroll['current_bankroll']:.2f}")

            with col2:
                st.metric("📊 Scommesse Pending", bankroll.get('pending_bets_count', 0))

            with col3:
                total_bets = bankroll.get('total_bets_count', 0)
                st.metric("📈 Totale Scommesse", total_bets)

            # Recent activity
            st.subheader("📋 Attività Recente")

            # Get recent bets
            recent_bets = db_manager.get_pending_bets()[:5]  # Last 5 pending bets

            if recent_bets:
                bets_data = []
                for bet in recent_bets:
                    bets_data.append({
                        'Game': f"{bet.away_team} @ {bet.home_team}",
                        'Type': f"{bet.bet_type} {bet.line}",
                        'Odds': bet.odds,
                        'Stake': f"€{bet.stake:.2f}",
                        'Date': bet.placed_at.strftime('%Y-%m-%d %H:%M')
                    })

                df_bets = pd.DataFrame(bets_data)
                st.dataframe(df_bets, use_container_width=True)
            else:
                st.info("Nessuna scommessa recente trovata.")

    except Exception as e:
        st.error(f"❌ Errore nel caricamento della dashboard: {e}")

def render_gestione_scommesse():
    """Sezione completa per la gestione delle scommesse."""
    st.header("💎 Gestione Completa Scommesse")

    try:
        # Initialize database manager
        db_manager = BettingDatabaseManager()

        # Get comprehensive bets data
        all_bets = db_manager.get_all_bets_comprehensive()
        bankroll_status = db_manager.get_bankroll_status()

        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "💰 Bankroll Attuale",
                f"€{bankroll_status['current_bankroll']:.2f}",
                delta=f"€{bankroll_status.get('total_profit_loss', 0):.2f}"
            )

        with col2:
            st.metric("📊 Pending Bets", len(all_bets.get('pending', [])))

        with col3:
            st.metric("✅ Settled Bets", len(all_bets.get('settled', [])))

        with col4:
            win_rate = bankroll_status.get('win_rate', 0)
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")

        # Tabs for different bet views
        tab1, tab2, tab3 = st.tabs(["📋 Pending Bets", "✅ Settled Bets", "📊 All Bets"])

        with tab1:
            render_pending_bets(all_bets.get('pending', []))

        with tab2:
            render_settled_bets(all_bets.get('settled', []))

        with tab3:
            render_all_bets(all_bets.get('all', []))

        # Close database connection
        db_manager.close()

    except Exception as e:
        st.error(f"❌ Errore nel caricamento delle scommesse: {e}")
        st.write("Dettagli errore:", str(e))

def render_pending_bets(pending_bets):
    """Render pending bets table."""
    st.subheader("📋 Scommesse Pending")

    if not pending_bets:
        st.info("Nessuna scommessa pending trovata.")
        return

    bets_data = []
    for bet in pending_bets:
        bets_data.append({
            'Game ID': bet.game_id,
            'Game': f"{bet.away_team} @ {bet.home_team}",
            'Type': f"{bet.bet_type} {bet.line}",
            'Odds': bet.odds,
            'Stake': f"€{bet.stake:.2f}",
            'Potential Win': f"€{bet.payout:.2f}",
            'Placed At': bet.placed_at.strftime('%Y-%m-%d %H:%M')
        })

    df = pd.DataFrame(bets_data)
    st.dataframe(df, use_container_width=True)

    # Summary stats
    total_staked = sum(bet.stake for bet in pending_bets)
    total_potential = sum(bet.payout for bet in pending_bets)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("💵 Totale Investito", f"€{total_staked:.2f}")
    with col2:
        st.metric("🎯 Potenziale Vincita", f"€{total_potential:.2f}")

def render_settled_bets(settled_bets):
    """Render settled bets table."""
    st.subheader("✅ Scommesse Concluse")

    if not settled_bets:
        st.info("Nessuna scommessa conclusa trovata.")
        return

    bets_data = []
    for bet in settled_bets:
        profit = bet.payout - bet.stake if bet.result == 'Won' else -bet.stake
        bets_data.append({
            'Game ID': bet.game_id,
            'Game': f"{bet.away_team} @ {bet.home_team}",
            'Type': f"{bet.bet_type} {bet.line}",
            'Odds': bet.odds,
            'Stake': f"€{bet.stake:.2f}",
            'Result': bet.result,
            'Profit/Loss': f"€{profit:.2f}",
            'Settled At': bet.settled_at.strftime('%Y-%m-%d %H:%M') if bet.settled_at else 'N/A'
        })

    df = pd.DataFrame(bets_data)
    st.dataframe(df, use_container_width=True)

    # Summary stats
    total_profit = sum((bet.payout - bet.stake) if bet.result == 'Won' else -bet.stake for bet in settled_bets)
    wins = sum(1 for bet in settled_bets if bet.result == 'Won')
    losses = sum(1 for bet in settled_bets if bet.result == 'Lost')
    win_rate = (wins / len(settled_bets) * 100) if settled_bets else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💵 Profit/Loss Totale", f"€{total_profit:.2f}")
    with col2:
        st.metric("🏆 Vittorie", wins)
    with col3:
        st.metric("❌ Sconfitte", losses)
    with col4:
        st.metric("🎯 Win Rate", f"{win_rate:.1f}%")

def render_all_bets(all_bets):
    """Render all bets table."""
    st.subheader("📊 Tutte le Scommesse")

    if not all_bets:
        st.info("Nessuna scommessa trovata.")
        return

    bets_data = []
    for bet in all_bets:
        status = "✅ Settled" if bet.result else "📋 Pending"
        profit_loss = ""
        if bet.result:
            profit_loss = f"€{(bet.payout - bet.stake):.2f}" if bet.result == 'Won' else f"-€{bet.stake:.2f}"

        bets_data.append({
            'Game ID': bet.game_id,
            'Game': f"{bet.away_team} @ {bet.home_team}",
            'Type': f"{bet.bet_type} {bet.line}",
            'Odds': bet.odds,
            'Stake': f"€{bet.stake:.2f}",
            'Status': status,
            'Result': bet.result or 'Pending',
            'P/L': profit_loss,
            'Placed At': bet.placed_at.strftime('%Y-%m-%d %H:%M')
        })

    df = pd.DataFrame(bets_data)
    st.dataframe(df, use_container_width=True)

def render_test_funzionalita():
    """Sezione per testare le funzionalità implementate."""
    st.header("🧪 Test Funzionalità Implementate")

    if st.button("🧪 Esegui Test Completo", type="primary"):
        with st.spinner("Esecuzione test in corso..."):
            try:
                # Test 1: Connessione database
                with BettingDatabaseManager() as db_manager:
                    st.success("✅ 1. Test connessione database manager")

                    # Test 2: Bankroll status
                    bankroll = db_manager.get_bankroll_status()
                    st.success(f"✅ 2. Bankroll status: €{bankroll['current_bankroll']:.2f}")

                    # Test 3: Comprehensive bets
                    all_bets = db_manager.get_all_bets_comprehensive()
                    pending_count = len(all_bets.get('pending', []))
                    settled_count = len(all_bets.get('settled', []))
                    st.success(f"✅ 3. Comprehensive bets: {pending_count} pending, {settled_count} settled")

                    # Test 4: Data store status
                    status = db_manager.get_data_store_status()
                    st.success(f"✅ 4. Data store status: {status.get('database_size_mb', 0):.2f} MB")

                st.success("🎉 Tutti i test passati con successo!")
                st.info("💡 Le nuove funzionalità sono operative e pronte all'uso!")

            except Exception as e:
                st.error(f"❌ Errore durante i test: {e}")
                st.write("Dettagli errore:", str(e))

    # Manual refresh button
    if st.button("🔄 Refresh Database Status"):
        with st.spinner("Aggiornamento in corso..."):
            try:
                with BettingDatabaseManager() as db_manager:
                    status = db_manager.sync_data_store()
                    st.success("✅ Database sincronizzato con successo!")
                    st.json(status)
            except Exception as e:
                st.error(f"❌ Errore sincronizzazione: {e}")

if __name__ == "__main__":
    main()