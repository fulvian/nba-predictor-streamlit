import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from nba_predictor.intelligence.bayesian_updater import BayesianUpdater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dynamic_impact():
    updater = BayesianUpdater()

    # Test Cases
    star_update = updater.update_prediction_with_items(
        baseline_mean=220.0,
        baseline_std=10.0,
        news_items=[{"type": "injury", "status": "Out", "player": "Nikola Jokic"}],
    )

    role_update = updater.update_prediction_with_items(
        baseline_mean=220.0,
        baseline_std=10.0,
        news_items=[
            {"type": "injury", "status": "Out", "player": "Kentavious Caldwell-Pope"}
        ],
    )

    star_impact = 220.0 - star_update.updated_score_dist[0]
    role_impact = 220.0 - role_update.updated_score_dist[0]

    logger.info(f"Star Impact (Jokic Out): {star_impact:.2f} pts")
    logger.info(f"Role Impact (KCP Out): {role_impact:.2f} pts")

    # Verification
    # Star impact (3.0 * -1.5 = -4.5) should be > Role impact (1.2 * -1.5 = -1.8)
    assert abs(star_impact) > abs(role_impact), (
        "Star impact should be greater than role player impact"
    )
    assert abs(star_impact) > 4.0, "Star impact should be significant (> 4.0)"

    logger.info("✅ Dynamic Impact Logic Verified Successfully")


if __name__ == "__main__":
    test_dynamic_impact()
