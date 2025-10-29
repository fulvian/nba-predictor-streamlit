"""
📊 Manual Odds Calculator - Sistema Legacy per Quote da Linea Centrale

Basato sul sistema legacy trovato in deprecated/probabilistic_model.py
Implementa lo schema fisso di quote utilizzato nel sistema originale.
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ManualOddsCalculator:
    """
    Calcolatore di quote basato su linea centrale inserita manualmente.

    Replica il sistema legacy con schema quote fisso:
    - 33 combinazioni di quote da -8.0 a +8.0 punti dalla linea centrale
    - Quote Over/Under predefinite per ogni offset
    - Linea centrale = 1.90/1.90 (50/50)
    """

    # Schema quote fisso dal sistema legacy
    QUOTE_SCHEMA = [
        (-8.0, 1.38, 2.85), (-7.5, 1.40, 2.75), (-7.0, 1.43, 2.65), (-6.5, 1.45, 2.60),
        (-6.0, 1.47, 2.55), (-5.5, 1.50, 2.50), (-5.0, 1.52, 2.40), (-4.5, 1.56, 2.35),
        (-4.0, 1.57, 2.30), (-3.5, 1.62, 2.25), (-3.0, 1.64, 2.20), (-2.5, 1.66, 2.15),
        (-2.0, 1.71, 2.10), (-1.5, 1.74, 2.05), (-1.0, 1.76, 2.00), (-0.5, 1.80, 1.96),
        (0.0, 1.90, 1.90),  # Linea centrale
        (0.5, 1.95, 1.80), (1.0, 2.00, 1.76), (1.5, 2.05, 1.74), (2.0, 2.10, 1.71),
        (2.5, 2.15, 1.66), (3.0, 2.20, 1.64), (3.5, 2.25, 1.62), (4.0, 2.30, 1.57),
        (4.5, 2.35, 1.55), (5.0, 2.40, 1.52), (5.5, 2.50, 1.50), (6.0, 2.55, 1.47),
        (6.5, 2.60, 1.45), (7.0, 2.65, 1.43), (7.5, 2.70, 1.41), (8.0, 2.85, 1.38)
    ]

    def __init__(self):
        """Inizializza il calcolatore di quote manuali."""
        self.logger = logging.getLogger(__name__)

    def generate_odds_from_central_line(self, central_line: float) -> List[Dict[str, Any]]:
        """
        Genera tutte le quote a partire dalla linea centrale inserita.

        Args:
            central_line: Linea centrale del bookmaker (es: 225.0)

        Returns:
            Lista di dizionari con quote generate
        """
        generated_odds = []

        for offset, over_quote, under_quote in self.QUOTE_SCHEMA:
            calculated_line = central_line + offset

            generated_odds.append({
                'line': calculated_line,
                'over_quote': over_quote,
                'under_quote': under_quote,
                'offset': offset,
                'bookmaker': 'Manual Input',
                'timestamp': None  # Non c'è timestamp per quote manuali
            })

        self.logger.info(f"✅ Generated {len(generated_odds)} odds from central line {central_line}")

        return generated_odds

    def get_best_betting_opportunities(self, central_line: float, predicted_total: float = None) -> Dict[str, Any]:
        """
        Identifica le migliori opportunità di betting basate su predizioni.

        Args:
            central_line: Linea centrale del bookmaker
            predicted_total: Predizione del totale punti (opzionale)

        Returns:
            Dizionario con le migliori opportunità di betting
        """
        odds = self.generate_odds_from_central_line(central_line)

        # Se non c'è una predizione, restituisci solo le quote
        if predicted_total is None:
            return {
                'central_line': central_line,
                'generated_odds': odds,
                'best_over': None,
                'best_under': None,
                'value_bets': []
            }

        # Calcola il valore per ogni quota basato sulla predizione
        value_bets = []

        for odd in odds:
            line = odd['line']

            # Calcolo valore per Over
            if line < predicted_total:
                # Predizione > Linea, Over è value bet
                implied_prob = 1 / odd['over_quote']
                predicted_prob = self._estimate_probability(predicted_total, line, 'over')
                edge = predicted_prob - implied_prob

                if edge > 0:  # Solo value positivi
                    value_bets.append({
                        'type': 'over',
                        'line': line,
                        'odds': odd['over_quote'],
                        'implied_probability': implied_prob,
                        'predicted_probability': predicted_prob,
                        'edge': edge,
                        'edge_percentage': edge * 100,
                        'line_difference': predicted_total - line
                    })

            # Calcolo valore per Under
            if line > predicted_total:
                # Predizione < Linea, Under è value bet
                implied_prob = 1 / odd['under_quote']
                predicted_prob = self._estimate_probability(predicted_total, line, 'under')
                edge = predicted_prob - implied_prob

                if edge > 0:  # Solo value positivi
                    value_bets.append({
                        'type': 'under',
                        'line': line,
                        'odds': odd['under_quote'],
                        'implied_probability': implied_prob,
                        'predicted_probability': predicted_prob,
                        'edge': edge,
                        'edge_percentage': edge * 100,
                        'line_difference': line - predicted_total
                    })

        # Ordina per edge decrescente
        value_bets.sort(key=lambda x: x['edge'], reverse=True)

        # Trova le migliori quote Over/Under
        best_over = None
        best_under = None

        if predicted_total is not None:
            # Trova la quota Over più vicina alla predizione
            over_odds = [odd for odd in odds if odd['line'] < predicted_total]
            if over_odds:
                best_over = min(over_odds, key=lambda x: abs(x['line'] - predicted_total))

            # Trova la quota Under più vicina alla predizione
            under_odds = [odd for odd in odds if odd['line'] > predicted_total]
            if under_odds:
                best_under = min(under_odds, key=lambda x: abs(x['line'] - predicted_total))

        return {
            'central_line': central_line,
            'predicted_total': predicted_total,
            'generated_odds': odds,
            'best_over': best_over,
            'best_under': best_under,
            'value_bets': value_bets,
            'total_value_opportunities': len(value_bets)
        }

    def _estimate_probability(self, predicted_total: float, line: float, bet_type: str) -> float:
        """
        Stima la probabilità basata sulla differenza tra predizione e linea.

        Basato sul sistema legacy con distribuzione normale modificata per betting NBA.

        Args:
            predicted_total: Predizione del totale punti
            line: Linea di scommessa
            bet_type: 'over' o 'under'

        Returns:
            Probabilità stimata (0-1)
        """
        # Differenza tra predizione e linea
        difference = predicted_total - line

        # Sistema di probabilità avanzato basato su legacy
        if bet_type == 'over':
            # Over è più probabile quanto più grande è la differenza positiva
            if difference >= 10:
                return 0.75  # 75% per differenze molto grandi
            elif difference >= 7:
                return 0.70  # 70% per differenze grandi
            elif difference >= 5:
                return 0.65  # 65% per differenze medie
            elif difference >= 3:
                return 0.60  # 60% per differenze moderate
            elif difference >= 1.5:
                return 0.55  # 55% per differenze piccole
            elif difference >= 0:
                return 0.52  # 52% minimo per essere value
            else:
                # Se la predizione è sotto la linea, Over è meno probabile
                return max(0.25, 0.52 + difference * 0.05)  # Decremento lineare
        else:  # under
            # Under è più probabile quanto più grande è la differenza negativa
            if difference <= -10:
                return 0.75  # 75% per differenze molto grandi
            elif difference <= -7:
                return 0.70  # 70% per differenze grandi
            elif difference <= -5:
                return 0.65  # 65% per differenze medie
            elif difference <= -3:
                return 0.60  # 60% per differenze moderate
            elif difference <= -1.5:
                return 0.55  # 55% per differenze piccole
            elif difference <= 0:
                return 0.52  # 52% minimo per essere value
            else:
                # Se la predizione è sopra la linea, Under è meno probabile
                return max(0.25, 0.52 - difference * 0.05)  # Decremento lineare

    def format_odds_for_display(self, odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formatta i dati delle quote per display nel dashboard.

        Args:
            odds_data: Dati grezzi delle quote

        Returns:
            Dati formattati per display
        """
        result = {
            'central_line': odds_data.get('central_line'),
            'predicted_total': odds_data.get('predicted_total'),
            'summary': f"Central Line: {odds_data.get('central_line')}",
            'status': 'Manual Input'
        }

        # Aggiungi migliori quote
        if odds_data.get('best_over'):
            result['best_over'] = {
                'line': odds_data['best_over']['line'],
                'odds': odds_data['best_over']['over_quote'],
                'type': 'Over'
            }

        if odds_data.get('best_under'):
            result['best_under'] = {
                'line': odds_data['best_under']['line'],
                'odds': odds_data['best_under']['under_quote'],
                'type': 'Under'
            }

        # Aggiungi opportunità di value betting
        if odds_data.get('value_bets'):
            top_value_bets = odds_data['value_bets'][:5]  # Top 5 value bets
            result['value_bets'] = []
            for bet in top_value_bets:
                result['value_bets'].append({
                    'type': bet['type'].title(),
                    'line': bet['line'],
                    'odds': bet['odds'],
                    'edge': f"{bet['edge_percentage']:.1f}%",
                    'line_diff': bet['line_difference']
                })

        return result

    def calculate_best_bet(self, central_line: float, predicted_total: float = None, bankroll: float = 1000.0) -> Dict[str, Any]:
        """
        Calcola la migliore scommessa usando l'algoritmo di ottimizzazione del sistema legacy.

        Implementa il sistema di scoring basato su edge, probabilità e quote ottimizzate.

        Args:
            central_line: Linea centrale del bookmaker
            predicted_total: Predizione del totale punti
            bankroll: Bankroll disponibile per betting

        Returns:
            Dizionario con la migliore scommessa e analisi completa
        """
        if predicted_total is None:
            return {
                'error': 'Predicted total is required for best bet calculation',
                'best_bet': None,
                'analysis': None
            }

        # Ottieni tutte le opportunità
        opportunities = self.get_best_betting_opportunities(central_line, predicted_total)
        value_bets = opportunities['value_bets']

        if not value_bets:
            return {
                'error': None,
                'best_bet': None,
                'analysis': {
                    'total_opportunities': 0,
                    'value_bets_found': 0,
                    'message': 'Nessuna value bet trovata. Provare con una diversa predizione.',
                    'all_opportunities': opportunities['generated_odds']
                }
            }

        # Calcolo scoring basato su sistema legacy
        scored_bets = []

        for bet in value_bets:
            edge = bet.get('edge', 0)
            probability = bet.get('predicted_probability', 0)
            odds = bet.get('odds', 1.0)

            if edge <= 0 or probability <= 0 or odds <= 1.0:
                continue

            # SCORING EDGE (30% weight)
            if edge >= 0.15:  # 15%+
                edge_score = 30
            elif edge >= 0.10:  # 10-15%
                edge_score = 25 + (edge - 0.10) * 100
            elif edge >= 0.05:  # 5-10%
                edge_score = 15 + (edge - 0.05) * 200
            elif edge >= 0.02:  # 2-5%
                edge_score = 5 + (edge - 0.02) * 333
            else:  # <2%
                edge_score = edge * 250

            # SCORING PROBABILITY (50% weight) - Sistema amplificato
            if probability > 0.65:
                prob_score = 35
            elif 0.60 <= probability <= 0.65:
                prob_score = 25 + (probability - 0.60) * 200
            elif 0.55 <= probability < 0.60:
                prob_score = 15 + (probability - 0.55) * 200
            elif 0.52 <= probability < 0.55:
                prob_score = 5 + (probability - 0.52) * 333
            else:  # 50-52% - Fascia critica
                prob_score = (probability - 0.50) * 250

            # SCORING ODDS (20% weight) - Sistema potenziato
            if 1.70 <= odds <= 1.95:
                odds_score = 30
            elif 1.60 <= odds < 1.70:
                odds_score = 18
            elif 1.95 < odds <= 2.10:
                odds_score = 20
            elif 2.10 < odds <= 2.30:
                odds_score = 12
            elif 2.30 < odds <= 2.60:
                odds_score = 8
            else:
                odds_score = max(3, 15 - abs(odds - 1.8) * 8)

            # Calcolo score totale
            total_score = (
                edge_score * 0.30 +      # Edge
                prob_score * 0.50 +      # Probabilità dominante
                odds_score * 0.20        # Quote potenziate
            )

            # Calcolo stake建议基于Kelly Criterion (modificato)
            kelly_fraction = (edge * probability) / (odds - 1)
            kelly_fraction = min(kelly_fraction, 0.25)  # Max 25% del bankroll

            # Stake basato su risk level
            if probability >= 0.70:
                risk_multiplier = 1.0  # Pieno
            elif probability >= 0.60:
                risk_multiplier = 0.8  # Conservativo
            else:
                risk_multiplier = 0.6  # Molto conservativo

            recommended_stake = bankroll * kelly_fraction * risk_multiplier
            recommended_stake = max(5.0, min(recommended_stake, bankroll * 0.10))  # Min €5, Max 10%

            bet_copy = bet.copy()
            bet_copy.update({
                'optimization_score': total_score,
                'edge_score': edge_score,
                'prob_score': prob_score,
                'odds_score': odds_score,
                'recommended_stake': recommended_stake,
                'kelly_fraction': kelly_fraction,
                'risk_multiplier': risk_multiplier,
                'potential_win': recommended_stake * (odds - 1),
                'risk_level': self._calculate_risk_level(probability),
                'confidence_level': self._calculate_confidence_level(total_score)
            })
            scored_bets.append(bet_copy)

        if not scored_bets:
            return {
                'error': None,
                'best_bet': None,
                'analysis': {
                    'total_opportunities': len(value_bets),
                    'value_bets_found': 0,
                    'message': 'Nessuna scommessa ha superato il filtro di qualità.',
                    'filtered_bets': value_bets
                }
            }

        # Trova la migliore scommessa
        scored_bets.sort(key=lambda x: x['optimization_score'], reverse=True)
        best_bet = scored_bets[0]

        return {
            'error': None,
            'best_bet': best_bet,
            'analysis': {
                'total_opportunities': len(opportunities['generated_odds']),
                'value_bets_found': len(value_bets),
                'qualified_bets': len(scored_bets),
                'best_score': best_bet['optimization_score'],
                'top_alternatives': scored_bets[1:4],  # Top 3 alternative
                'all_scored_bets': scored_bets
            }
        }

    def _calculate_risk_level(self, probability: float) -> str:
        """
        Calcola il livello di rischio basato sulla probabilità.

        Args:
            probability: Probabilità di successo (0-1)

        Returns:
            Stringa con livello di rischio
        """
        if probability >= 0.70:
            return "🟢 BASSO"
        elif probability >= 0.60:
            return "🟡 MEDIO"
        else:
            return "🔴 ALTO"

    def _calculate_confidence_level(self, score: float) -> str:
        """
        Calcola il livello di confidenza basato sullo score di ottimizzazione.

        Args:
            score: Score di ottimizzazione (0-100)

        Returns:
            Stringa con livello di confidenza
        """
        if score >= 80:
            return "🔥 ALTA"
        elif score >= 60:
            return "⚡ MEDIA"
        else:
            return "⚪ BASSA"

    def generate_comprehensive_analysis(self, central_line: float, predicted_total: float = None, bankroll: float = 1000.0) -> Dict[str, Any]:
        """
        Genera un'analisi completa basata sul sistema legacy.

        Include: migliori opportunità, best bet, risk management, e stack management.

        Args:
            central_line: Linea centrale del bookmaker
            predicted_total: Predizione del totale punti
            bankroll: Bankroll disponibile

        Returns:
            Analisi completa del betting
        """
        # Analisi base
        basic_analysis = self.get_best_betting_opportunities(central_line, predicted_total)

        # Best bet analysis
        best_bet_analysis = self.calculate_best_bet(central_line, predicted_total, bankroll)

        # Statistiche aggregate
        all_odds = basic_analysis['generated_odds']
        value_bets = basic_analysis['value_bets']

        # Distribution analysis
        over_lines = [odd for odd in all_odds if odd['line'] < (predicted_total or central_line)]
        under_lines = [odd for odd in all_odds if odd['line'] > (predicted_total or central_line)]

        summary = {
            'central_line': central_line,
            'predicted_total': predicted_total,
            'line_difference': (predicted_total or central_line) - central_line if predicted_total else 0,
            'total_odds_generated': len(all_odds),
            'value_opportunities': len(value_bets),
            'over_opportunities': len([b for b in value_bets if b['type'] == 'over']),
            'under_opportunities': len([b for b in value_bets if b['type'] == 'under']),
            'best_bet': best_bet_analysis.get('best_bet'),
            'analysis': best_bet_analysis.get('analysis'),
            'bankroll': bankroll,
            'recommended_total_exposure': self._calculate_total_exposure(bankroll, len(value_bets))
        }

        return {
            'summary': summary,
            'basic_analysis': basic_analysis,
            'best_bet_analysis': best_bet_analysis,
            'risk_management': {
                'max_stake_per_bet': bankroll * 0.10,
                'recommended_total_exposure': summary['recommended_total_exposure'],
                'diversification': min(5, len(value_bets)) if value_bets else 0
            },
            'value_bets_distribution': {
                'by_edge': self._group_by_edge(value_bets),
                'by_probability': self._group_by_probability(value_bets),
                'by_odds_range': self._group_by_odds_range(value_bets)
            }
        }

    def _calculate_total_exposure(self, bankroll: float, opportunities: int) -> float:
        """
        Calcola l'esposizione totale raccomandata.

        Args:
            bankroll: Bankroll totale
            opportunities: Numero di opportunità

        Returns:
            Esposizione totale raccomandata
        """
        if opportunities == 0:
            return 0.0

        # Massimo 25% del bankroll o 5% per scommessa
        max_per_bet = bankroll * 0.05
        max_total = bankroll * 0.25

        return min(max_total, max_per_bet * min(opportunities, 5))

    def _group_by_edge(self, value_bets: List[Dict]) -> Dict[str, int]:
        """Raggruppa value bets per range di edge."""
        groups = {'0-2%': 0, '2-5%': 0, '5-10%': 0, '10%+': 0}
        for bet in value_bets:
            edge_pct = bet.get('edge_percentage', 0)
            if edge_pct < 2:
                groups['0-2%'] += 1
            elif edge_pct < 5:
                groups['2-5%'] += 1
            elif edge_pct < 10:
                groups['5-10%'] += 1
            else:
                groups['10%+'] += 1
        return groups

    def _group_by_probability(self, value_bets: List[Dict]) -> Dict[str, int]:
        """Raggruppa value bets per range di probabilità."""
        groups = {'50-55%': 0, '55-60%': 0, '60-65%': 0, '65%+': 0}
        for bet in value_bets:
            prob = bet.get('predicted_probability', 0) * 100
            if prob < 55:
                groups['50-55%'] += 1
            elif prob < 60:
                groups['55-60%'] += 1
            elif prob < 65:
                groups['60-65%'] += 1
            else:
                groups['65%+'] += 1
        return groups

    def _group_by_odds_range(self, value_bets: List[Dict]) -> Dict[str, int]:
        """Raggruppa value bets per range di quote."""
        groups = {'<1.70': 0, '1.70-1.95': 0, '1.95-2.10': 0, '2.10+': 0}
        for bet in value_bets:
            odds = bet.get('odds', 0)
            if odds < 1.70:
                groups['<1.70'] += 1
            elif odds <= 1.95:
                groups['1.70-1.95'] += 1
            elif odds <= 2.10:
                groups['1.95-2.10'] += 1
            else:
                groups['2.10+'] += 1
        return groups


# Singleton instance per uso in tutto il sistema
_manual_odds_calculator = ManualOddsCalculator()