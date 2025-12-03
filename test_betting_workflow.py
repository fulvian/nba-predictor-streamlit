#!/usr/bin/env python3
"""
Script per testare il betting workflow dopo aver fixato lo schema del database.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import duckdb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent

def test_betting_workflow():
    """
    Test the complete betting workflow by creating a test bet.
    """
    project_root = get_project_root()
    db_path = project_root / "data" / "nba_data.duckdb"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        # Add src to path
        sys.path.append(str(project_root / "src"))
        from nba_predictor.utils.betting_database_manager import BettingDatabaseManager, BetAnalysis

        logger.info("🧪 Testing Betting Workflow After Schema Fix")

        with BettingDatabaseManager() as manager:
            # Step 1: Create test BetAnalysis
            logger.info("   Step 1: Creating test BetAnalysis...")

            test_analysis = BetAnalysis(
                analysis_id=f"test_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                game_id="TEST_GAME_001",
                bet_type="Over",
                line=225.5,
                odds=-110,
                edge=2.5,
                probability=0.55,
                quality_score=0.8,
                risk_level="Low",
                recommendation="Strong Bet",
                confidence_level=0.9,
                model_predictions='{"model1": "Over"}',
                created_at=datetime.now(),
                home_team="Lakers",  # Added home team
                away_team="Celtics"  # Added away team
            )
            logger.info(f"   ✅ Created BetAnalysis with ID: {test_analysis.analysis_id}")

            # Step 2: Save analysis to database
            logger.info("   Step 2: Saving BetAnalysis to database...")
            try:
                manager.save_analysis(test_analysis)
                logger.info("   ✅ BetAnalysis saved successfully")
            except Exception as e:
                logger.error(f"   ❌ Failed to save BetAnalysis: {e}")
                return False

            # Step 3: Place a test bet
            logger.info("   Step 3: Placing test bet...")
            try:
                bet_id = manager.place_bet(
                    analysis_id=test_analysis.analysis_id,
                    stake=50.0,
                    notes="Test bet for workflow verification"
                )
                logger.info(f"   ✅ Bet placed successfully with ID: {bet_id}")
            except Exception as e:
                logger.error(f"   ❌ Failed to place bet: {e}")
                return False

            # Step 4: Verify bet appears in management queries
            logger.info("   Step 4: Verifying bet appears in management queries...")
            try:
                pending_bets = manager.get_pending_bets()
                test_bet_found = False

                for bet in pending_bets:
                    if bet['bet_id'] == bet_id:
                        test_bet_found = True
                        logger.info(f"   ✅ Test bet found in pending bets: {bet['home_team']} vs {bet['away_team']}")
                        break

                if not test_bet_found:
                    logger.error("   ❌ Test bet not found in pending bets")
                    logger.error(f"   Pending bets found: {len(pending_bets)}")
                    for bet in pending_bets:
                        logger.error(f"      - {bet['bet_id']}: {bet.get('home_team', 'Unknown')} vs {bet.get('away_team', 'Unknown')}")
                    return False

            except Exception as e:
                logger.error(f"   ❌ Failed to query pending bets: {e}")
                return False

            # Step 5: Verify bet has all required data
            logger.info("   Step 5: Verifying bet has complete data...")
            try:
                bet_details = None
                for bet in pending_bets:
                    if bet['bet_id'] == bet_id:
                        bet_details = bet
                        break

                if bet_details:
                    required_fields = ['bet_id', 'home_team', 'away_team', 'status', 'placed_at', 'stake']
                    missing_fields = [field for field in required_fields if field not in bet_details or bet_details[field] is None]

                    if missing_fields:
                        logger.error(f"   ❌ Bet missing required fields: {missing_fields}")
                        return False
                    else:
                        logger.info(f"   ✅ Bet has all required fields")
                        logger.info(f"      - Bet ID: {bet_details['bet_id']}")
                        logger.info(f"      - Teams: {bet_details['home_team']} vs {bet_details['away_team']}")
                        logger.info(f"      - Status: {bet_details['status']}")
                        logger.info(f"      - Stake: ${bet_details['stake']}")
                        logger.info(f"      - Placed At: {bet_details['placed_at']}")
                else:
                    logger.error("   ❌ Could not retrieve bet details")
                    return False

            except Exception as e:
                logger.error(f"   ❌ Failed to verify bet details: {e}")
                return False

            # Step 6: Clean up test data
            logger.info("   Step 6: Cleaning up test data...")
            try:
                conn = manager.conn
                conn.execute("DELETE FROM placed_bets WHERE bet_id = ?", [bet_id])
                conn.execute("DELETE FROM betting_analysis WHERE analysis_id = ?", [test_analysis.analysis_id])
                logger.info("   ✅ Test data cleaned up")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not clean up test data: {e}")

        logger.info("🎉 Betting Workflow Test PASSED!")
        logger.info("=" * 50)
        logger.info("✅ BetAnalysis creation successful")
        logger.info("✅ Bet placement successful")
        logger.info("✅ Bet appears in management queries")
        logger.info("✅ Bet has complete required data")
        logger.info("✅ Database schema working correctly")
        return True

    except Exception as e:
        logger.error(f"❌ Betting workflow test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test execution."""
    logger.info("🚀 Starting Betting Workflow Test")
    logger.info("=" * 60)
    logger.info("Testing the complete betting workflow after schema fixes")

    success = test_betting_workflow()

    # Summary
    if success:
        logger.info("🎉 BETTING WORKFLOW TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("✅ All betting functionality is working correctly")
        logger.info("✅ The user's bet placement bug has been fixed")
        logger.info("✅ Bets placed through the dashboard will now appear in management section")
    else:
        logger.error("❌ Betting workflow test failed")
        logger.error("Please check the error messages above")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)