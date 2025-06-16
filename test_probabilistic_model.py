import sys
sys.path.insert(0, '.')

from probabilistic_model_v2 import ProbabilisticModel

def test_prediction():
    model = ProbabilisticModel()
    
    # Dati di esempio per una partita
    game_data = {
        'team_stats': {
            'home': {
                'ORtg': 112.3,
                'DRtg': 108.7,
                'Pace': 100.2,
                'eFG_PCT': 0.54,
                'FT_RATE': 0.28,
                'win_streak': 3
            },
            'away': {
                'ORtg': 110.5,
                'DRtg': 107.9,
                'Pace': 98.7,
                'eFG_PCT': 0.52,
                'FT_RATE': 0.24,
                'win_streak': 2
            }
        },
        'injury_reports': {
            'impact_analysis': {
                'home_team_impact': -2.1,
                'away_team_impact': -1.3
            }
        },
        'momentum_impact': {
            'home': 1.2,
            'away': 0.8
        }
    }
    
    prediction = model.predict_score_with_injuries_and_momentum(game_data)
    print("Risultato della previsione:", prediction)

if __name__ == "__main__":
    test_prediction()
