"""
NBA Totals Service - High-level API for managing Over/Under odds.

This service provides a unified interface for:
- Updating historic totals from Kaggle and OddsHarvester
- Updating upcoming totals for daily monitoring
- Querying odds by game, season, or bookmaker
- Computing closing odds approximations
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import polars as pl

from ..etl.odds.migrate_kaggle_totals import migrate_kaggle_totals
from ..etl.odds.normalize_odds_harvester import (
    OddsHarvesterNormalizer,
    extract_season_from_date,
)
from ..models.nba_totals_odds import NbaTotalsOddsRepository, OddsSource
from ..scraping.odds_harvester_runner import OddsHarvesterRunner, OddsHarvesterConfig

logger = logging.getLogger(__name__)


class NbaTotalsService:
    """
    High-level service for managing NBA Over/Under odds data.

    Provides unified access to historical and live odds from multiple sources:
    - Kaggle dataset (historical, 2011-2018)
    - OddsHarvester (2020+, live scraping)
    - The Odds API (real-time, if configured)
    """

    # Bookmaker priority order
    BOOKMAKER_PRIORITY = [
        "Bet365",  # User preference
        "Pinnacle",  # Sharp bookmaker
        "DraftKings",  # US major
        "FanDuel",  # US major
        "BetMGM",  # US major
        "average",  # Fallback
    ]

    def __init__(
        self,
        db_path: Path = Path("data/nba_betting.duckdb"),
        kaggle_path: Optional[Path] = None,
        odds_harvester_path: Optional[Path] = None,
        priority_bookmaker: str = "Pinnacle",
    ) -> None:
        """
        Initialize service.

        Args:
            db_path: Path to DuckDB database
            kaggle_path: Path to Kaggle totals CSV
            odds_harvester_path: Path to OddsHarvester repo
            priority_bookmaker: Primary bookmaker for closing odds
        """
        self.db_path = Path(db_path)
        self.kaggle_path = kaggle_path or Path(
            "data/nba_odds_csv/nba_betting_totals.csv"
        )
        self.games_path = Path("data/nba_odds_csv/nba_games_all.csv")

        # Initialize repository
        self.repository = NbaTotalsOddsRepository(db_path)
        self.repository.initialize_schema()

        # Initialize normalizer
        self.normalizer = OddsHarvesterNormalizer(
            priority_bookmaker=priority_bookmaker,
            fallback_to_average=True,
        )

        # Initialize runner (optional, for live scraping)
        self.runner = OddsHarvesterRunner(
            odds_harvester_path=odds_harvester_path,
            config=OddsHarvesterConfig(
                markets=["over/under"],
                headless=True,
            ),
        )

        self.priority_bookmaker = priority_bookmaker

    def update_historic_totals(
        self,
        season_start: str = "2020-2021",
        season_end: str = "current",
        force_reimport: bool = False,
    ) -> dict[str, int]:
        """
        Update database with historic odds from all sources.

        Workflow:
        1. Check what's already in DB
        2. Migrate Kaggle data if not present (limited to 2018)
        3. Run OddsHarvester for missing seasons (2020+)
        4. Return summary of imported records

        Args:
            season_start: First season to import (e.g., "2020-2021")
            season_end: Last season to import ("current" = ongoing)
            force_reimport: If True, reimport all data

        Returns:
            Dict with counts: {"kaggle": N, "scraped": M, "total": N+M}
        """
        result = {"kaggle": 0, "scraped": 0, "total": 0}

        # Check existing data
        existing_counts = self.repository.count_records(by_source=True)
        logger.info(f"Existing records by source: {existing_counts}")

        # Step 1: Migrate Kaggle data if available and not already imported
        if self.kaggle_path.exists() and (
            force_reimport or existing_counts.get("kaggle", 0) == 0
        ):
            try:
                kaggle_df = migrate_kaggle_totals(
                    kaggle_path=self.kaggle_path,
                    games_path=self.games_path if self.games_path.exists() else None,
                    min_season="2011-2012",  # Kaggle data starts ~2011
                    max_season="2018-2019",  # Kaggle data ends ~2018
                )

                if not kaggle_df.is_empty():
                    inserted = self.repository.insert_odds(kaggle_df)
                    result["kaggle"] = inserted
                    logger.info(f"Migrated {inserted} Kaggle records")
            except Exception as e:
                logger.error(f"Failed to migrate Kaggle data: {e}")

        # Step 2: Check if we need to scrape more recent seasons
        # Convert season strings to list
        current_year = datetime.now().year
        if season_end == "current":
            # NBA season is Oct-June, so current season depends on month
            if datetime.now().month >= 10:
                end_year = current_year
            else:
                end_year = current_year - 1
        else:
            end_year = int(season_end.split("-")[0])

        start_year = int(season_start.split("-")[0])

        # Seasons to potentially scrape (2020 onwards where Kaggle doesn't cover)
        seasons_to_scrape = []
        for year in range(max(start_year, 2020), end_year + 1):
            season_str = f"{year}-{year + 1}"
            # Check if season already has data
            season_count = len(self.repository.get_odds_by_season(season_str))
            if season_count == 0 or force_reimport:
                seasons_to_scrape.append(season_str)

        logger.info(f"Seasons needing scrape: {seasons_to_scrape}")

        # Step 3: Scrape missing seasons (if OddsHarvester is available)
        for season in seasons_to_scrape:
            try:
                output_file = self.runner.run_historic_scrape(
                    season=season,
                    dry_run=False,
                )

                if output_file and output_file.exists():
                    scraped_df = self.normalizer.normalize_to_dataframe(output_file)
                    if not scraped_df.is_empty():
                        inserted = self.repository.insert_odds(scraped_df)
                        result["scraped"] += inserted
                        logger.info(
                            f"Imported {inserted} records from OddsHarvester for {season}"
                        )
            except Exception as e:
                logger.warning(f"Could not scrape season {season}: {e}")

        result["total"] = result["kaggle"] + result["scraped"]

        # Log final state
        final_counts = self.repository.count_records(by_source=True)
        logger.info(f"Final records by source: {final_counts}")

        return result

    def update_upcoming_totals(self, days_ahead: int = 7) -> int:
        """
        Update odds for upcoming games.

        Args:
            days_ahead: Number of days ahead to scrape

        Returns:
            Number of records imported
        """
        try:
            output_file = self.runner.run_upcoming_scrape(days_ahead=days_ahead)

            if output_file and output_file.exists():
                scraped_df = self.normalizer.normalize_to_dataframe(
                    output_file,
                    scrape_datetime=datetime.utcnow(),
                )

                if not scraped_df.is_empty():
                    inserted = self.repository.insert_odds(scraped_df)
                    logger.info(f"Imported {inserted} upcoming odds records")
                    return inserted

            return 0

        except Exception as e:
            logger.error(f"Failed to update upcoming totals: {e}")
            return 0

    def get_game_totals(
        self,
        game_id: str,
        bookmaker: Optional[str] = None,
        closing_only: bool = True,
    ) -> pl.DataFrame:
        """
        Get odds for a specific game.

        Args:
            game_id: NBA game ID
            bookmaker: Optional bookmaker filter (default: priority bookmaker)
            closing_only: If True, return only closing odds

        Returns:
            DataFrame with odds records
        """
        book = bookmaker or self.priority_bookmaker

        df = self.repository.get_odds_by_game(
            game_id=game_id,
            bookmaker=book,
            closing_only=closing_only,
        )

        # If priority bookmaker not found, try average
        if df.is_empty() and not bookmaker:
            df = self.repository.get_odds_by_game(
                game_id=game_id,
                closing_only=closing_only,
            )

            if not df.is_empty():
                df = self._compute_average_odds(df)

        return df

    def get_season_totals(
        self,
        season: str,
        bookmaker: Optional[str] = None,
        closing_only: bool = True,
    ) -> pl.DataFrame:
        """
        Get all odds for a season.

        Args:
            season: Season string (e.g., "2023-2024")
            bookmaker: Optional bookmaker filter
            closing_only: If True, return only closing odds

        Returns:
            DataFrame with odds records
        """
        return self.repository.get_odds_by_season(
            season=season,
            bookmaker=bookmaker or self.priority_bookmaker,
            closing_only=closing_only,
        )

    def select_closing_totals(
        self,
        odds_df: pl.DataFrame,
        schedule_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Identify closing odds (last odds before game start).

        Strategy:
        1. Join odds with schedule on game_id to get game_datetime
        2. Filter: scrape_datetime < game_datetime
        3. For each (game_id, bookmaker), take max(scrape_datetime)
        4. If no game_datetime, take max(scrape_datetime) overall
        5. Mark is_closing = True

        Args:
            odds_df: DataFrame with odds records (game_id, bookmaker, scrape_datetime, ...)
            schedule_df: DataFrame with schedule (game_id, game_datetime)

        Returns:
            DataFrame with closing odds only, is_closing = True
        """
        if odds_df.is_empty():
            return odds_df

        # Prepare schedule for join
        if "game_datetime" not in schedule_df.columns:
            # Try to construct from game_date + game_time
            if (
                "game_date" in schedule_df.columns
                and "game_time" in schedule_df.columns
            ):
                schedule_df = schedule_df.with_columns(
                    (pl.col("game_date") + " " + pl.col("game_time")).alias(
                        "game_datetime"
                    )
                )
            elif "game_date" in schedule_df.columns:
                # Use end of game day as default (23:59)
                schedule_df = schedule_df.with_columns(
                    (pl.col("game_date") + " 23:59:00").alias("game_datetime")
                )
            else:
                logger.warning(
                    "Schedule missing game_datetime, using latest odds as closing"
                )
                # Fallback: just use latest odds per game/bookmaker
                return self._select_latest_as_closing(odds_df)

        schedule_df = schedule_df.select(
            [
                pl.col("game_id").cast(pl.Utf8),
                pl.col("game_datetime").str.strptime(
                    pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False
                ),
            ]
        ).drop_nulls()

        # Join odds with schedule
        joined = odds_df.join(
            schedule_df,
            on="game_id",
            how="left",
        )

        # Filter odds before game time (where game_datetime is not null)
        # Or keep all (where game_datetime is null)
        pre_game = joined.filter(
            (pl.col("game_datetime").is_null())
            | (pl.col("scrape_datetime") < pl.col("game_datetime"))
        )

        # Select latest per game + bookmaker
        closing = (
            pre_game.sort("scrape_datetime", descending=True)
            .group_by(["game_id", "bookmaker"])
            .first()
            .with_columns(pl.lit(True).alias("is_closing"))
            .drop("game_datetime")
        )

        return closing

    def _select_latest_as_closing(self, odds_df: pl.DataFrame) -> pl.DataFrame:
        """Select latest odds per game/bookmaker as closing."""
        return (
            odds_df.sort("scrape_datetime", descending=True)
            .group_by(["game_id", "bookmaker"])
            .first()
            .with_columns(pl.lit(True).alias("is_closing"))
        )

    def _compute_average_odds(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute average odds across all bookmakers."""
        return df.group_by(
            [
                "game_id",
                "game_date",
                "season",
                "stage",
                "home_team_id",
                "away_team_id",
                "total_points_line",
            ]
        ).agg(
            [
                pl.col("odds_over_decimal").mean(),
                pl.col("odds_under_decimal").mean(),
                pl.col("scrape_datetime").max(),
                pl.lit("average").alias("bookmaker"),
                pl.first("source"),
                pl.first("is_closing"),
            ]
        )

    def get_ml_features(
        self,
        season: str,
        bookmaker: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Get odds data formatted as ML features.

        Features generated:
        - total_line: Central total points line
        - implied_prob_over: 1 / odds_over
        - implied_prob_under: 1 / odds_under
        - vig: implied_over + implied_under - 1
        - odds_ratio: odds_over / odds_under
        - log_odds_ratio: log(odds_over / odds_under)

        Args:
            season: Season to get features for
            bookmaker: Preferred bookmaker (default: priority)

        Returns:
            DataFrame with ML features per game
        """
        df = self.get_season_totals(
            season=season,
            bookmaker=bookmaker,
            closing_only=True,
        )

        if df.is_empty():
            return df

        # Compute ML features
        features = df.with_columns(
            [
                pl.col("total_points_line").alias("total_line"),
                # Implied probabilities
                (1.0 / pl.col("odds_over_decimal")).alias("implied_prob_over"),
                (1.0 / pl.col("odds_under_decimal")).alias("implied_prob_under"),
                # Vig (overround)
                (
                    (1.0 / pl.col("odds_over_decimal"))
                    + (1.0 / pl.col("odds_under_decimal"))
                    - 1.0
                ).alias("vig"),
                # Odds ratio
                (pl.col("odds_over_decimal") / pl.col("odds_under_decimal")).alias(
                    "odds_ratio"
                ),
                # Log odds ratio
                (pl.col("odds_over_decimal") / pl.col("odds_under_decimal"))
                .log()
                .alias("log_odds_ratio"),
            ]
        )

        # Select relevant columns for ML
        ml_columns = [
            "game_id",
            "game_date",
            "season",
            "home_team_id",
            "away_team_id",
            "total_line",
            "odds_over_decimal",
            "odds_under_decimal",
            "implied_prob_over",
            "implied_prob_under",
            "vig",
            "odds_ratio",
            "log_odds_ratio",
            "bookmaker",
            "source",
        ]

        return features.select([c for c in ml_columns if c in features.columns])

    def get_statistics(self) -> dict[str, Any]:
        """Get summary statistics about the odds database."""
        by_source = self.repository.count_records(by_source=True)
        bookmakers = self.repository.get_available_bookmakers()
        seasons = self.repository.get_available_seasons()

        return {
            "total_records": sum(by_source.values()),
            "records_by_source": by_source,
            "bookmakers": bookmakers,
            "seasons": seasons,
            "priority_bookmaker": self.priority_bookmaker,
        }

    def close(self) -> None:
        """Close database connection."""
        self.repository.close()


# Convenience function for quick access
def get_totals_service(db_path: Optional[Path] = None) -> NbaTotalsService:
    """Get a configured NbaTotalsService instance."""
    return NbaTotalsService(
        db_path=db_path or Path("data/nba_betting.duckdb"),
        priority_bookmaker="Pinnacle",
    )


if __name__ == "__main__":
    # Test service
    logging.basicConfig(level=logging.INFO)

    service = get_totals_service()

    # Update with Kaggle data
    result = service.update_historic_totals(
        season_start="2011-2012",
        season_end="2018-2019",
    )

    print(f"\nImport result: {result}")
    print(f"\nStatistics: {service.get_statistics()}")

    service.close()
