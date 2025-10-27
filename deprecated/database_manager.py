# database_manager.py

import sqlalchemy
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, DateTime, MetaData
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///market_movements.db')

engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Tabella per lo storico delle quote
odds_history = Table('odds_history', metadata,
    Column('id', Integer, primary_key=True),
    Column('game_id', String, nullable=False, index=True),
    Column('sport_key', String, nullable=False),
    Column('home_team', String, nullable=False),
    Column('away_team', String, nullable=False),
    Column('bookmaker', String, nullable=False),
    Column('market_key', String, nullable=False),
    Column('line', Float, nullable=False),
    Column('odds_over', Float, nullable=False),
    Column('odds_under', Float, nullable=False),
    Column('api_source', String, nullable=True),  # 'theoddsapi', 'therundown', 'apisports'
    Column('timestamp', DateTime, default=datetime.utcnow)
)

# NUOVA Tabella per lo storico del consenso (Public/Sharp Money)
betting_consensus_history = Table('betting_consensus_history', metadata,
    Column('id', Integer, primary_key=True),
    Column('game_id', String, nullable=False, index=True),
    Column('source', String, nullable=False), # Es. 'SportsInsights', 'VSIN', 'ActionNetwork'
    Column('public_tickets_percentage_over', Float, nullable=True),
    Column('public_money_percentage_over', Float, nullable=True),
    Column('sharp_money_indicator', String, nullable=True), # Es. 'Steam Move on Over', 'RLM on Under'
    Column('reverse_line_movement_detected', String, nullable=True), # 'Yes', 'No', dettagli
    Column('steam_move_detected', String, nullable=True), # Informazioni sui steam moves
    Column('line_at_time', Float, nullable=True), # Linea al momento del rilevamento
    Column('timestamp', DateTime, default=datetime.utcnow)
)

def initialize_database():
    """Crea entrambe le tabelle se non esistono."""
    metadata.create_all(engine)
    print("Database initialized with tables: odds_history, betting_consensus_history.")

def save_data(data: list, table: Table):
    """Funzione generica per salvare dati in una tabella specifica."""
    if not data:
        return
    with engine.connect() as conn:
        conn.execute(table.insert(), data)
        print(f"Saved {len(data)} records to table {table.name}.")

def get_history_for_game(game_id: str, table: Table) -> pd.DataFrame:
    """Funzione generica per recuperare lo storico da una tabella."""
    query = sqlalchemy.select(table).where(table.c.game_id == game_id).order_by(table.c.timestamp.asc())
    df = pd.read_sql(query, engine)
    return df

# Helper specifici per le quote
def save_odds_data(data: list):
    """Salva dati delle quote nel database."""
    save_data(data, odds_history)

def get_odds_history_for_game(game_id: str, bookmaker: str = None) -> pd.DataFrame:
    """
    Recupera lo storico delle quote per una specifica partita.
    Se bookmaker è specificato, filtra per quel bookmaker.
    """
    query = sqlalchemy.select(odds_history).where(odds_history.c.game_id == game_id)
    
    if bookmaker:
        query = query.where(odds_history.c.bookmaker == bookmaker)
    
    query = query.order_by(odds_history.c.timestamp.asc())
    df = pd.read_sql(query, engine)
    return df

# Helper specifici per i dati di consenso
def save_consensus_data(data: list):
    """Salva dati di consenso nel database."""
    save_data(data, betting_consensus_history)

def get_consensus_history(game_id: str) -> pd.DataFrame:
    """Recupera lo storico del consenso per una partita specifica."""
    return get_history_for_game(game_id, betting_consensus_history)

def get_all_active_games() -> list:
    """Restituisce una lista di tutti i game_id attivi nel database."""
    query = sqlalchemy.select(odds_history.c.game_id).distinct()
    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result.fetchall()]

if __name__ == '__main__':
    initialize_database() 