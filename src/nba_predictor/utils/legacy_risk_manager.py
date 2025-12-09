#!/usr/bin/env python3
"""
🎯 LEGACY RISK MANAGER - Replica Esatta del Sistema Legacy

Questo modulo replica ESATTAMENTE il sistema di gestione del rischio,
bankroll e stack management dal sistema legacy.

Caratteristiche:
- Replica esatta dell'algoritmo _calculate_advanced_stake
- Sistema completo di bankroll management
- Quality score multi-fattoriale identico al legacy
- Gestione scommesse pendenti e storico
- Limiti di esposizione e Kelly Criterion

Creato per replicare esattamente il comportamento del sistema legacy
in main_backup.py e probabilistic_model.py
"""

import json
import os
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


class LegacyRiskManager:
    """
    🎯 GESTIONE RISCHIO LEGACY - Replica Esatta del Sistema Originale

    Implementa tutti gli algoritmi esatti del sistema legacy:
    - _calculate_advanced_stake: Calcolo stake ultra-granulare
    - _calculate_quality_score: Quality score multi-fattoriale
    - Bankroll management con persistenza JSON
    - Gestione scommesse pendenti
    - Kelly Criterion e limiti di esposizione
    """

    # Schema quote identico al sistema legacy
    QUOTE_SCHEMA = [
        (-8.0, 1.38, 2.85),
        (-7.5, 1.40, 2.75),
        (-7.0, 1.43, 2.65),
        (-6.5, 1.45, 2.60),
        (-6.0, 1.48, 2.55),
        (-5.5, 1.50, 2.50),
        (-5.0, 1.53, 2.45),
        (-4.5, 1.55, 2.40),
        (-4.0, 1.57, 2.35),
        (-3.5, 1.60, 2.30),
        (-3.0, 1.62, 2.28),
        (-2.5, 1.65, 2.25),
        (-2.0, 1.68, 2.20),
        (-1.5, 1.70, 2.15),
        (-1.0, 1.73, 2.10),
        (-0.5, 1.75, 2.05),
        (0.0, 1.90, 1.90),  # Linea centrale
        (0.5, 2.05, 1.75),
        (1.0, 2.10, 1.73),
        (1.5, 2.15, 1.70),
        (2.0, 2.20, 1.68),
        (2.5, 2.25, 1.65),
        (3.0, 2.28, 1.62),
        (3.5, 2.30, 1.60),
        (4.0, 2.35, 1.57),
        (4.5, 2.40, 1.55),
        (5.0, 2.45, 1.53),
        (5.5, 2.50, 1.50),
        (6.0, 2.55, 1.48),
        (6.5, 2.60, 1.45),
        (7.0, 2.65, 1.43),
        (7.5, 2.75, 1.40),
        (8.0, 2.85, 1.38),
    ]

    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        self.bankroll_file = os.path.join(data_path, "bankroll.json")
        self.pending_bets_file = os.path.join(data_path, "pending_bets.json")
        self.bet_history_file = os.path.join(data_path, "bet_history.json")

        # Inizializza directory e dati
        self._ensure_data_directory()
        self.current_bankroll = self._load_bankroll()

    def _ensure_data_directory(self):
        """Assicura che le directory dati esistano."""
        os.makedirs(self.data_path, exist_ok=True)

    def _load_bankroll(self, default: float = 100.0) -> float:
        """
        Carica il bankroll dal file JSON (identico al legacy).

        Args:
            default: Valore di default se nessun file esiste

        Returns:
            Bankroll attuale
        """
        bankroll_paths = [self.bankroll_file, "bankroll.json"]

        for bankroll_path in bankroll_paths:
            try:
                if os.path.exists(bankroll_path):
                    with open(bankroll_path, "r") as f:
                        data = json.load(f)
                        bankroll_value = float(data.get("current_bankroll", default))
                        print(
                            f"💰 Bankroll caricato da {bankroll_path}: €{bankroll_value:.2f}"
                        )
                        return bankroll_value
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        print(f"ℹ️ Nessun file bankroll trovato. Usando valore default: €{default}")
        self._save_bankroll(default)
        return default

    def _save_bankroll(self, new_bankroll: float):
        """
        Salva il bankroll aggiornato (identico al legacy).

        Args:
            new_bankroll: Nuovo valore del bankroll
        """
        try:
            bankroll_data = {"current_bankroll": float(new_bankroll)}

            # Salva nel file principale
            with open(self.bankroll_file, "w") as f:
                json.dump(bankroll_data, f, indent=2)

            # Salva anche nel backup
            backup_path = os.path.join(self.data_path, "bankroll_backup.json")
            with open(backup_path, "w") as f:
                json.dump(bankroll_data, f, indent=2)

            self.current_bankroll = new_bankroll
            print(f"💰 Bankroll aggiornato e salvato: €{new_bankroll:.2f}")

        except Exception as e:
            print(f"⚠️ Errore nel salvataggio del bankroll: {e}")

    def update_bankroll_from_bet(
        self, bet_result: Dict[str, Any], actual_total: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Aggiorna il bankroll basandosi sul risultato di una scommessa (identico al legacy).

        Args:
            bet_result: Dict con 'type', 'line', 'odds', 'stake'
            actual_total: Punteggio totale reale della partita

        Returns:
            Risultato dell'aggiornamento o None
        """
        if not actual_total or not bet_result:
            return None

        bet_type = bet_result.get("type")
        line = bet_result.get("line")
        odds = bet_result.get("odds")
        stake = bet_result.get("stake")

        if not all([bet_type, line, odds, stake]):
            print("⚠️ Informazioni scommessa incomplete per aggiornamento bankroll")
            return None

        # Determina se la scommessa è vinta
        if bet_type == "OVER":
            bet_won = actual_total > line
        else:  # UNDER
            bet_won = actual_total <= line

        # Calcola profit/loss
        if bet_won:
            profit = stake * (odds - 1)
            new_bankroll = self.current_bankroll + profit
            print(f"🟢 SCOMMESSA VINTA! Profit: €{profit:.2f}")
        else:
            loss = stake
            new_bankroll = self.current_bankroll - loss
            print(f"🔴 Scommessa persa. Loss: €{loss:.2f}")

        # Salva il nuovo bankroll
        self._save_bankroll(new_bankroll)

        return {
            "bet_won": bet_won,
            "profit_loss": profit if bet_won else -stake,
            "new_bankroll": new_bankroll,
        }

    def calculate_advanced_stake(
        self,
        edge: float,
        estimated_prob: float,
        odds: float,
        bankroll: Optional[float] = None,
        quality_data: Optional[Dict] = None,
        bet_direction: Optional[str] = None,  # Kept for backward compatibility
        kelly_fraction: float = 0.5,  # Half-Kelly = industry standard (Ziemba, Thorp)
    ) -> float:
        """
        🎯 KELLY CRITERION STAKE CALCULATOR - Industry Standard Implementation

        Based on academic literature (Ziemba, MacLean, Thorp) and professional standards:
        - Formula: f* = (b*p - q) / b where b = odds-1, p = win prob, q = 1-p
        - Default: Half-Kelly (50%) as recommended by professionals
        - Max stake: 5% of bankroll (industry standard upper bound)
        - Min stake: 1% of bankroll or €1.00

        References:
        - "The Kelly Capital Growth Investment Criterion" (MacLean, Thorp, Ziemba)
        - "Fortune's Formula" (Poundstone)
        - Pinnacle betting guidelines

        Parameters:
        - edge: Mathematical edge (not used directly - Kelly uses prob & odds)
        - estimated_prob: Estimated win probability (0.0-1.0)
        - odds: Decimal bookmaker odds (e.g., 1.90)
        - bankroll: Available bankroll (uses current if None)
        - quality_data: Quality data (optional, for compatibility)
        - kelly_fraction: Fraction of Kelly to bet (default 0.5 = Half-Kelly)

        Returns:
            Recommended stake in Euro
        """
        # Use current bankroll if not specified
        if bankroll is None:
            bankroll = self.current_bankroll

        # === KELLY CRITERION FORMULA ===
        # f* = (b*p - q) / b
        # where b = odds - 1, p = estimated_prob, q = 1 - p

        b = odds - 1.0  # Net odds
        p = estimated_prob
        q = 1.0 - p

        # Full Kelly fraction
        if b > 0:
            full_kelly = (b * p - q) / b
        else:
            full_kelly = 0.0

        # Ensure non-negative (no bet if negative edge)
        full_kelly = max(0.0, full_kelly)

        # === APPLY FRACTIONAL KELLY ===
        # Industry standard: Half-Kelly (0.5) reduces volatility significantly
        # while retaining ~75% of the growth rate (Ziemba et al.)
        fractional_kelly = full_kelly * kelly_fraction

        # === STAKE LIMITS (Industry Standards) ===
        # Min: 1% bankroll or €1.00 (whichever is greater)
        min_stake = max(1.0, bankroll * 0.01)

        # Max: 5% bankroll (professional upper bound)
        max_stake = bankroll * 0.05

        # Calculate raw stake
        raw_stake = bankroll * fractional_kelly

        # === NO BET CONDITIONS ===
        # 1. Probability too low (< 50% = no value expected)
        # 2. Kelly suggests 0 or negative
        if estimated_prob < 0.50 or fractional_kelly <= 0:
            return 0.0

        # Apply limits
        final_stake = min(raw_stake, max_stake)
        final_stake = max(final_stake, min_stake)

        # Round appropriately
        if final_stake >= 10:
            return round(final_stake, 0)
        elif final_stake >= 1:
            return round(final_stake, 1)
        else:
            return round(final_stake, 2)

    def calculate_mean_variance_kelly_stake(
        self,
        estimated_prob: float,
        prob_std_error: float,
        odds: float,
        bankroll: float = None,
        max_stake_pct: float = 0.05,
        min_stake_pct: float = 0.01,
        beta: float = 0.10,  # 10% Kelly = very conservative, produces ~5x stake variability
        sigma_target: float = 0.15,  # Target uncertainty for full confidence (calibrated for typical model sigma 12-15)
    ) -> float:
        """
        🎯 Confidence-Weighted Kelly (Industry Standard)

        Formula: f = β × c² × f_Kelly

        Where:
        - f_Kelly = (b×p - q) / b  (standard Kelly)
        - β = 0.5 (Half-Kelly, conservative) or 0.25 (Quarter-Kelly, very conservative)
        - c = confidence score derived from σ_p: c = min(1, σ_target / σ_p)
        - c² gives stronger haircut when confidence decreases

        This formula:
        - Produces realistic stakes (1-5% of bankroll)
        - Incorporates model uncertainty correctly
        - Is used by professional betting syndicates

        References:
        - Perplexity research on professional stake sizing
        - Industry standard fractional Kelly with confidence weighting

        Parameters:
        - estimated_prob: Model's win probability (0.0-1.0)
        - prob_std_error: Standard error of probability (σ_p)
        - odds: Decimal bookmaker odds (e.g., 1.90)
        - bankroll: Available bankroll (uses current if None)
        - max_stake_pct: Maximum stake as % of bankroll (default 5%)
        - min_stake_pct: Minimum stake as % of bankroll (default 1%)
        - beta: Fractional Kelly parameter (default 0.5 = Half-Kelly)
        - sigma_target: Target σ for full confidence (default 0.10)

        Returns:
            Recommended stake in Euro
        """
        if bankroll is None:
            bankroll = self.current_bankroll

        # Validate inputs
        if estimated_prob <= 0.50 or odds <= 1.0:
            return 0.0

        b = odds - 1.0  # Net odds
        p = estimated_prob
        q = 1.0 - p

        # === STEP 1: Calculate Standard Kelly ===
        # f_Kelly = (b×p - q) / b
        f_kelly = (b * p - q) / b

        # No bet if Kelly is negative (no edge)
        if f_kelly <= 0:
            return 0.0

        # === STEP 2: Calculate Confidence Score from σ_p ===
        # c = min(1, σ_target / σ_p)
        # Higher σ_p → lower confidence → lower stake
        if prob_std_error <= 0:
            prob_std_error = 0.15  # Default: moderate uncertainty

        confidence = min(1.0, sigma_target / prob_std_error)

        # === STEP 3: Apply Confidence-Weighted Kelly ===
        # f = β × c² × f_Kelly
        # c² gives stronger haircut when confidence is low
        f_star = beta * (confidence**2) * f_kelly

        # === STEP 4: Apply Limits ===
        min_stake = max(1.0, bankroll * min_stake_pct)
        max_stake = bankroll * max_stake_pct

        raw_stake = bankroll * f_star

        # If stake too small, don't bet
        if raw_stake < min_stake * 0.5:
            return 0.0

        final_stake = min(raw_stake, max_stake)
        final_stake = max(final_stake, min_stake)

        # Round appropriately
        if final_stake >= 10:
            return round(final_stake, 0)
        elif final_stake >= 1:
            return round(final_stake, 1)
        else:
            return round(final_stake, 2)

    def calculate_quality_score(
        self, edge: float, estimated_prob: float, odds: float
    ) -> Dict[str, Any]:
        """
        Calcola un punteggio di qualità avanzato (replica esatta legacy).

        NUOVO ALGORITMO MULTI-FATTORIALE:
        - Edge Score: Normalizza il vantaggio matematico
        - Confidence Score: Valuta la fiducia nella predizione
        - Risk Score: Analizza il profilo rischio/rendimento
        - Consistency Score: Premia la consistenza del modello
        - Final Quality: Combinazione pesata con scaling intelligente
        """

        # 1. EDGE SCORE - Normalizza il vantaggio matematico (0-100)
        edge_pct = edge * 100  # Converte in percentuale
        if edge_pct <= 0:
            edge_score = 0
        elif edge_pct >= 20:  # Edge > 20% = punteggio massimo
            edge_score = 100
        else:
            # Scala non lineare: premia edge alti
            edge_score = (edge_pct / 20) ** 0.7 * 100

        # 2. CONFIDENCE SCORE - Valuta la fiducia nella predizione (0-100)
        prob_pct = estimated_prob * 100
        if prob_pct < 45:
            confidence_score = 0  # Troppo incerto
        elif prob_pct > 95:
            confidence_score = 30  # Troppo estremo, probabilmente errore
        elif 50 <= prob_pct <= 65:
            confidence_score = 100  # Sweet spot: fiducia alta ma realistica
        elif 65 < prob_pct <= 75:
            confidence_score = 90  # Molto buono
        elif 75 < prob_pct <= 85:
            confidence_score = 75  # Buono ma più rischioso
        elif 45 <= prob_pct < 50:
            confidence_score = 40  # Limite accettabile
        else:
            confidence_score = 60  # Media

        # 3. RISK SCORE - Analisi profilo rischio/rendimento (0-100)
        # Calcolo Kelly Criterion
        kelly_fraction = 0
        if odds > 1 and estimated_prob > 0:
            # Kelly: (p*b - q) / b dove b = odds-1, p = prob, q = 1-p
            b = odds - 1
            q = 1 - estimated_prob
            kelly_fraction = (estimated_prob * b - q) / b
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Limita tra 0-25%

        # Risk score basato su Kelly e volatilità
        if kelly_fraction >= 0.10:
            risk_score = 100  # Ottimo
        elif kelly_fraction >= 0.05:
            risk_score = 85  # Buono
        elif kelly_fraction >= 0.02:
            risk_score = 70  # Accettabile
        elif kelly_fraction >= 0.01:
            risk_score = 50  # Moderato
        else:
            risk_score = 25  # Rischioso

        # 4. CONSISTENCY SCORE - Misura coerenza predizione vs mercato (0-100)
        # Differenza tra probabilità stimata e implicita
        implied_prob = 1 / odds
        prob_diff = abs(estimated_prob - implied_prob)
        if prob_diff <= 0.05:  # Differenza ≤ 5%
            consistency_score = 100
        elif prob_diff <= 0.10:  # Differenza ≤ 10%
            consistency_score = 80
        elif prob_diff <= 0.15:  # Differenza ≤ 15%
            consistency_score = 60
        elif prob_diff <= 0.20:  # Differenza ≤ 20%
            consistency_score = 40
        else:
            consistency_score = 20

        # 5. CALCOLO FINALE - Combinazione pesata con scaling intelligente
        # Pesi: Edge (40%), Confidence (30%), Risk (20%), Consistency (10%)
        raw_score = (
            edge_score * 0.40
            + confidence_score * 0.30
            + risk_score * 0.20
            + consistency_score * 0.10
        )

        # Scaling finale: trasforma da 0-100 a 0-1 con curva non lineare
        if raw_score >= 80:
            final_quality = 0.8 + (raw_score - 80) / 20 * 0.2  # 80-100 → 0.8-1.0
        elif raw_score >= 60:
            final_quality = 0.5 + (raw_score - 60) / 20 * 0.3  # 60-80 → 0.5-0.8
        elif raw_score >= 40:
            final_quality = 0.2 + (raw_score - 40) / 20 * 0.3  # 40-60 → 0.2-0.5
        else:
            final_quality = raw_score / 40 * 0.2  # 0-40 → 0.0-0.2

        return {
            "quality_score": final_quality,
            "edge_score": edge_score / 100,
            "confidence_score": confidence_score / 100,
            "risk_score": risk_score / 100,
            "consistency_score": consistency_score / 100,
            "raw_score": raw_score,
            "kelly_fraction": kelly_fraction,
            "edge": edge,
            "estimated_prob": estimated_prob,
            "implied_prob": implied_prob,
        }

    def calculate_risk_score(
        self, edge: float, estimated_prob: float, odds: float
    ) -> float:
        """
        Calcola il punteggio di rischio isolato seguendo pattern Context7 enterprise.

        Metodo Context7-compliant che:
        - Implementa calcolo rischio robusto e validato
        - Fornisce graceful degradation per input anomali
        - Include validazione input sanitization
        - Applica scaling intelligente con bounds checking
        - Supporta traceability per audit compliance

        Args:
            edge: Vantaggio matematico calcolato
            estimated_prob: Probabilità stimata dal modello
            odds: Quota decimal offerta

        Returns:
            float: Risk score normalizzato 0-1 con Context7 compliance
        """
        try:
            # Context7: Input sanitization e validation
            if not all(
                isinstance(x, (int, float)) for x in [edge, estimated_prob, odds]
            ):
                return 0.0

            if odds <= 1.0 or estimated_prob <= 0 or estimated_prob >= 1:
                return 0.0

            # Context7: Kelly Criterion calculation per risk assessment
            b = odds - 1  # Net odds
            q = 1 - estimated_prob

            # Context7: Protected calculation con bounds checking
            if b <= 0:
                return 0.0

            kelly_fraction = (estimated_prob * b - q) / b
            kelly_fraction = max(
                0, min(0.25, kelly_fraction)
            )  # Context7: Safety bounds

            # Context7: Risk multi-factor assessment
            # 1. Kelly-based risk (inverse relationship)
            kelly_risk = 1.0 - (kelly_fraction / 0.25)  # Higher kelly = lower risk

            # 2. Edge-based risk assessment
            edge_pct = edge * 100
            if edge_pct <= 0:
                edge_risk = 1.0  # Maximum risk se non c'è edge
            elif edge_pct >= 15:
                edge_risk = 0.0  # Minimum risk con edge elevato
            else:
                # Context7: Non-linear scaling per risk assessment
                edge_risk = 1.0 - (edge_pct / 15) ** 0.8

            # 3. Probability-based risk assessment
            prob_pct = estimated_prob * 100
            if prob_pct < 40 or prob_pct > 90:
                prob_risk = 1.0  # High risk per probabilità estreme
            elif 50 <= prob_pct <= 70:
                prob_risk = 0.0  # Minimum risk nel sweet spot
            elif 45 <= prob_pct < 50 or 70 < prob_pct <= 80:
                prob_risk = 0.3  # Low-moderate risk
            else:
                prob_risk = 0.6  # Moderate risk

            # Context7: Intelligent weighted combination
            # Pesi dinamici basati sulla confidence del modello
            confidence_weight = min(
                1.0, estimated_prob * 1.5
            )  # Higher confidence = more weight

            risk_score = (
                kelly_risk * 0.4  # 40% Kelly-based risk
                + edge_risk * 0.35  # 35% Edge-based risk
                + prob_risk * 0.25  # 25% Probability-based risk
            ) * confidence_weight + (
                1 - confidence_weight
            ) * 0.5  # Context7: Confidence adjustment

            # Context7: Final bounds checking e normalization
            risk_score = max(0.0, min(1.0, risk_score))

            # Context7: Traceability log per audit compliance
            if hasattr(self, "debug_mode") and self.debug_mode:
                print(f"🔍 Context7 Risk Score Calculation:")
                print(f"   Edge: {edge:.4f} → Edge Risk: {edge_risk:.3f}")
                print(f"   Est Prob: {estimated_prob:.4f} → Prob Risk: {prob_risk:.3f}")
                print(
                    f"   Odds: {odds:.3f} → Kelly: {kelly_fraction:.3f} → Kelly Risk: {kelly_risk:.3f}"
                )
                print(f"   Final Risk Score: {risk_score:.3f}")

            return risk_score

        except Exception as e:
            # Context7: Graceful degradation con logging strutturato
            print(f"⚠️ Context7 Risk Score Calculation Error: {str(e)}")
            print(f"📍 Input: edge={edge}, prob={estimated_prob}, odds={odds}")
            return 0.0  # Context7: Safe fallback per robustezza enterprise

    def generate_odds_from_central_line(
        self, central_line: float
    ) -> List[Dict[str, float]]:
        """
        Genera quote da linea centrale (identico al legacy).

        Args:
            central_line: Linea centrale del bookmaker

        Returns:
            Lista di quote generate
        """
        generated_odds = []
        for offset, over_quote, under_quote in self.QUOTE_SCHEMA:
            generated_odds.append(
                {
                    "line": central_line + offset,
                    "over_quote": over_quote,
                    "under_quote": under_quote,
                }
            )
        print(
            f"✅ Generate {len(generated_odds)} linee di quota attorno alla linea centrale {central_line}"
        )
        return generated_odds

    def analyze_betting_opportunities(
        self,
        distribution: Dict[str, Any],
        odds_list: List[Dict] = None,
        central_line: float = None,
        bankroll: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Analisi completa opportunità di betting (replica esatta legacy).

        Args:
            distribution: Risultati modello probabilistico
            odds_list: Lista quote disponibili
            central_line: Linea centrale per generazione quote
            bankroll: Bankroll disponibile

        Returns:
            Lista di opportunità analizzate
        """
        if bankroll is None:
            bankroll = self.current_bankroll

        # Se non ci sono quote, generale dalla linea centrale
        if not odds_list and central_line:
            odds_list = self.generate_odds_from_central_line(central_line)
        elif not odds_list:
            print("⚠️ Nessuna quota disponibile per l'analisi")
            return []

        predicted_total = distribution.get("predicted_mu", 0)
        confidence_sigma = distribution.get("predicted_sigma", 0)

        all_lines_analysis = []

        for odds_item in odds_list:
            line = odds_item["line"]
            over_odds = odds_item.get("over_quote", odds_item.get("odds", 1.90))
            under_odds = odds_item.get("under_quote", 1.90)

            # Calcola probabilità dalla distribuzione normale
            from scipy.stats import norm

            prob_over = 1 - norm.cdf(line, predicted_total, confidence_sigma)
            prob_under = norm.cdf(line, predicted_total, confidence_sigma)

            # Calcola edge per OVER
            implied_prob_over = 1 / over_odds
            true_prob_over = prob_over
            edge_over = true_prob_over - implied_prob_over

            # Calcola edge per UNDER
            implied_prob_under = 1 / under_odds
            true_prob_under = prob_under
            edge_under = true_prob_under - implied_prob_under

            # Quality scores
            quality_over = self.calculate_quality_score(edge_over, prob_over, over_odds)
            quality_under = self.calculate_quality_score(
                edge_under, prob_under, under_odds
            )

            # Valuta se sono VALUE bets
            is_value_over = edge_over > 0 and prob_over >= 0.50
            is_value_under = edge_under > 0 and prob_under >= 0.50

            # === CALCOLO σ_p (incertezza probabilità) ===
            # σ_p = standard error normalizzato da punti a probabilità
            # Più alto il σ del modello in punti, più alta l'incertezza sulla prob
            # Tipicamente: σ=10 punti → bassa incertezza, σ=20 punti → alta incertezza
            # Normalizziamo σ_p tra 0.05 e 0.25 per la formula f=β×c²×f_Kelly
            sigma_p = max(0.05, min(0.25, confidence_sigma / 100.0))

            # Calcola stake usando Mean-Variance Kelly (Thorp, MacLean, Ziemba 2011)
            stake_over = (
                self.calculate_mean_variance_kelly_stake(
                    prob_over, sigma_p, over_odds, bankroll
                )
                if is_value_over
                else 0
            )
            stake_under = (
                self.calculate_mean_variance_kelly_stake(
                    prob_under, sigma_p, under_odds, bankroll
                )
                if is_value_under
                else 0
            )

            all_lines_analysis.extend(
                [
                    {
                        "type": "OVER",
                        "line": line,
                        "odds": over_odds,
                        "probability": prob_over,
                        "implied_probability": implied_prob_over,
                        "true_probability": true_prob_over,
                        "edge": edge_over,
                        "quality_score": quality_over["quality_score"],
                        "edge_score": quality_over["edge_score"],
                        "confidence_score": quality_over["confidence_score"],
                        "risk_score": quality_over["risk_score"],
                        "consistency_score": quality_over["consistency_score"],
                        "stake": stake_over,
                        "is_value": is_value_over,
                        "kelly_fraction": quality_over["kelly_fraction"],
                    },
                    {
                        "type": "UNDER",
                        "line": line,
                        "odds": under_odds,
                        "probability": prob_under,
                        "implied_probability": implied_prob_under,
                        "true_probability": true_prob_under,
                        "edge": edge_under,
                        "quality_score": quality_under["quality_score"],
                        "edge_score": quality_under["edge_score"],
                        "confidence_score": quality_under["confidence_score"],
                        "risk_score": quality_under["risk_score"],
                        "consistency_score": quality_under["consistency_score"],
                        "stake": stake_under,
                        "is_value": is_value_under,
                        "kelly_fraction": quality_under["kelly_fraction"],
                    },
                ]
            )

        # Ordina per quality_score invece che per edge
        return sorted(
            all_lines_analysis, key=lambda x: x["quality_score"], reverse=True
        )

    def calculate_optimal_bet(
        self, opportunities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Calcola la scommessa ottimale usando algoritmo legacy.

        Args:
            opportunities: Lista di opportunità analizzate

        Returns:
            Scommessa ottimale o None
        """
        try:
            # SOGLIA EDGE PIÙ ALTA: Minimo 1% per essere considerato VALUE
            value_bets = [
                opp
                for opp in opportunities
                if opp.get("edge", 0) > 0.01 and opp.get("probability", 0) >= 0.50
            ]
            if not value_bets:
                return None

            scored_bets = []

            for bet in value_bets:
                edge = bet.get("edge", 0)
                probability = bet.get("probability", 0)
                odds = bet.get("odds", 1.0)

                if edge <= 0 or probability <= 0 or odds <= 1.0:
                    continue

                # Sistema di scoring legacy - Edge 30% + Probability 50% + Odds 20%
                # Edge score
                if edge >= 0.15:  # 15%+
                    edge_score = 30
                elif edge >= 0.10:  # 10-15%
                    edge_score = 25 + (edge - 0.10) * 100  # Scala 25-30
                elif edge >= 0.05:  # 5-10%
                    edge_score = 15 + (edge - 0.05) * 200  # Scala 15-25
                elif edge >= 0.02:  # 2-5%
                    edge_score = 5 + (edge - 0.02) * 333  # Scala 5-15
                else:  # <2%
                    edge_score = edge * 250  # Max 5 punti per edge molto bassi

                # Probability score
                if probability > 0.65:
                    prob_score = 35  # Bonus per probabilità molto alte
                elif 0.60 <= probability <= 0.65:
                    prob_score = 25 + (probability - 0.60) * 200  # Scala 25-35
                elif 0.55 <= probability < 0.60:
                    prob_score = 15 + (probability - 0.55) * 200  # Scala 15-25
                elif 0.52 <= probability < 0.55:
                    prob_score = (
                        5 + (probability - 0.52) * 333
                    )  # Scala 5-15 (AMPLIFICATA)
                else:  # 50-52% - FASCIA CRITICA
                    prob_score = (probability - 0.50) * 250  # 0-5 punti (MOLTO RIPIDA)

                # Odds score
                if 1.70 <= odds <= 1.95:
                    odds_score = 30  # Range ottimale massimo premio
                elif 1.60 <= odds < 1.70:
                    odds_score = 18  # Buono ma margine basso
                elif 1.95 < odds <= 2.10:
                    odds_score = 20  # Ancora accettabile
                elif 2.10 < odds <= 2.30:
                    odds_score = 12  # Rischio moderato
                elif 2.30 < odds <= 2.60:
                    odds_score = 8  # Rischio alto
                else:
                    odds_score = max(
                        3, 15 - abs(odds - 1.8) * 8
                    )  # Penalizzazione severa

                # Sistema pulito - Solo 3 componenti indipendenti
                total_score = (
                    edge_score * 0.30  # Edge
                    + prob_score * 0.50  # Probabilità dominante
                    + odds_score * 0.20  # Quote potenziate
                )

                bet_copy = bet.copy()
                normalized_score = total_score  # Già su scala 0-100

                bet_copy.update(
                    {
                        "optimization_score": normalized_score,
                        "edge_score": edge_score,
                        "prob_score": prob_score,
                        "odds_score": odds_score,
                        "total_raw_score": total_score,
                    }
                )
                scored_bets.append(bet_copy)

            if not scored_bets:
                return None

            best_bet = max(scored_bets, key=lambda x: x["optimization_score"])
            return best_bet

        except Exception as e:
            print(f"⚠️ Errore nel calcolo scommessa ottimale: {e}")
            return None

    def save_pending_bet(self, bet_data: Dict[str, Any], game_id: str) -> bool:
        """
        Salva una scommessa in attesa di risultato (identico al legacy).

        Args:
            bet_data: Dati della scommessa
            game_id: ID della partita

        Returns:
            True se salvato con successo
        """
        try:
            # Carica scommesse esistenti
            pending_bets = self._load_pending_bets()

            # Converte i dati in tipi JSON serializzabili
            clean_bet_data = {}
            for key, value in bet_data.items():
                if isinstance(value, (int, float, str)):
                    clean_bet_data[key] = value
                elif hasattr(value, "item"):  # NumPy scalars
                    clean_bet_data[key] = value.item()
                elif isinstance(value, bool):
                    clean_bet_data[key] = bool(value)
                else:
                    clean_bet_data[key] = float(value) if value is not None else 0.0

            # Controlla se esiste già una scommessa per questo game_id
            existing_bet_index = None
            for i, bet in enumerate(pending_bets):
                if bet.get("game_id") == game_id and bet.get("status") == "pending":
                    existing_bet_index = i
                    break

            if existing_bet_index is not None:
                # Sostituisci la scommessa esistente
                old_bet_data = pending_bets[existing_bet_index]["bet_data"]
                print(f"⚠️ SCOMMESSA ESISTENTE TROVATA per {game_id}:")
                print(
                    f"   ATTUALE: {old_bet_data.get('type', 'N/A')} {old_bet_data.get('line', 'N/A')} @ {old_bet_data.get('odds', 'N/A')} (€{old_bet_data.get('stake', 0):.2f})"
                )
                print(
                    f"   NUOVA:   {clean_bet_data.get('type', 'N/A')} {clean_bet_data.get('line', 'N/A')} @ {clean_bet_data.get('odds', 'N/A')} (€{clean_bet_data.get('stake', 0):.2f})"
                )

                pending_bets[existing_bet_index] = {
                    "bet_id": f"{game_id}_{clean_bet_data['type']}_{clean_bet_data['line']}",
                    "game_id": game_id,
                    "bet_data": clean_bet_data,
                    "timestamp": datetime.now().isoformat(),
                    "status": "pending",
                    "replaced_at": datetime.now().isoformat(),
                    "original_bet": old_bet_data,
                }
                print(
                    f"🔄 Scommessa sostituita: {clean_bet_data['type']} {clean_bet_data['line']}"
                )
            else:
                # Nessuna scommessa esistente, aggiungi normalmente
                pending_bet = {
                    "bet_id": f"{game_id}_{clean_bet_data['type']}_{clean_bet_data['line']}",
                    "game_id": game_id,
                    "bet_data": clean_bet_data,
                    "timestamp": datetime.now().isoformat(),
                    "status": "pending",
                }
                pending_bets.append(pending_bet)
                print(
                    f"💾 Scommessa salvata in attesa di risultato: {clean_bet_data['type']} {clean_bet_data['line']}"
                )

            # Salva le scommesse aggiornate
            with open(self.pending_bets_file, "w") as f:
                json.dump(pending_bets, f, indent=2)

            return True

        except Exception as e:
            print(f"⚠️ Errore nel salvataggio scommessa pendente: {e}")
            return False

    def _load_pending_bets(self) -> List[Dict[str, Any]]:
        """Carica le scommesse pendenti dal file."""
        try:
            if os.path.exists(self.pending_bets_file):
                with open(self.pending_bets_file, "r") as f:
                    return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return []

    def get_bankroll_status(self) -> Dict[str, Any]:
        """
        Ottiene lo stato completo del bankroll.

        Returns:
            Dizionario con stato bankroll e statistiche
        """
        pending_bets = self._load_pending_bets()
        pending_stake = sum(
            bet.get("bet_data", {}).get("stake", 0)
            for bet in pending_bets
            if bet.get("status") == "pending"
        )

        return {
            "current_bankroll": self.current_bankroll,
            "pending_stake": pending_stake,
            "available_bankroll": self.current_bankroll - pending_stake,
            "pending_bets_count": len(
                [b for b in pending_bets if b.get("status") == "pending"]
            ),
            "total_bets_count": len(pending_bets),
        }

    def assess_risk_level(self, bet: Dict[str, Any]) -> str:
        """
        Valuta il livello di rischio complessivo basato sul Quality Score legacy.

        Args:
            bet: Dati scommessa con quality_score

        Returns:
            Livello di rischio testuale
        """
        quality = bet.get("quality_score", 0)
        risk_score = bet.get("risk_score", 0)

        # Combinazione di quality score e risk score specifico
        if quality >= 0.8 and risk_score >= 0.8:
            return "MOLTO BASSO"
        elif quality >= 0.6 and risk_score >= 0.6:
            return "BASSO"
        elif quality >= 0.4 and risk_score >= 0.4:
            return "MODERATO"
        elif quality >= 0.2:
            return "ALTO"
        else:
            return "MOLTO ALTO"


def main():
    """Test del LegacyRiskManager"""
    print("🎯 Test Legacy Risk Manager")
    print("=" * 50)

    # Inizializza
    risk_manager = LegacyRiskManager()

    # Test calcolo quality score
    edge = 0.08  # 8%
    prob = 0.55  # 55%
    odds = 2.10

    quality = risk_manager.calculate_quality_score(edge, prob, odds)
    print(f"Quality Score: {quality['quality_score']:.3f}")
    print(f"Edge Score: {quality['edge_score']:.3f}")
    print(f"Confidence Score: {quality['confidence_score']:.3f}")
    print(f"Risk Score: {quality['risk_score']:.3f}")

    # Test calcolo stake
    stake = risk_manager.calculate_advanced_stake(edge, prob, odds)
    print(f"Stake raccomandato: €{stake:.2f}")

    # Test generazione quote
    odds = risk_manager.generate_odds_from_central_line(225.0)
    print(f"Generate {len(odds)} quote")

    # Stato bankroll
    status = risk_manager.get_bankroll_status()
    print(f"Bankroll attuale: €{status['current_bankroll']:.2f}")


if __name__ == "__main__":
    main()
