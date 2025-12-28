import unittest
from unittest.mock import MagicMock, patch
from nba_predictor.intelligence.odds_portal_scraper import OddsPortalScraper


class TestOddsPortalScraper(unittest.TestCase):
    def setUp(self):
        self.scraper = OddsPortalScraper(headless=True)

    def tearDown(self):
        self.scraper.close()

    def test_get_season_results_urls(self):
        """Test URL generation for multiple seasons."""
        urls = self.scraper.get_season_results_urls(2021, 2023)
        expected = [
            "https://www.oddsportal.com/basketball/usa/nba-2021-2022/results/",
            "https://www.oddsportal.com/basketball/usa/nba-2022-2023/results/",
        ]
        self.assertEqual(urls, expected)

    @patch("src.nba_predictor.intelligence.odds_portal_scraper.webdriver.Chrome")
    def test_init_driver(self, mock_chrome):
        """Test that the driver is initialized with correct options."""
        self.scraper._init_driver()
        mock_chrome.assert_called_once()
        self.assertIsNotNone(self.scraper.driver)

    def test_output_dir_exists(self):
        """Test that the output directory is created."""
        import os

        self.assertTrue(os.path.exists("data/raw/odds_portal"))


if __name__ == "__main__":
    unittest.main()
