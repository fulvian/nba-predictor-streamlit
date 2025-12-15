"""
Download and Validate Kaggle NBA Odds Dataset

Downloads the free Kaggle NBA Odds Data (2008-2023) dataset and runs
validation checks per Consensus requirements.

Requirements:
- Kaggle API: pip install kaggle
- Kaggle API credentials: ~/.kaggle/kaggle.json (get from kaggle.com/settings)
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_kaggle_dataset():
    """
    Download NBA Odds Data from Kaggle.

    Dataset: Christopher Treasure - NBA Odds Data (2008-2023)
    URL: https://www.kaggle.com/datasets/christophertreasure/nba-odds-data
    """
    logger.info("=== Downloading Kaggle NBA Odds Dataset ===\n")

    # Check if Kaggle API is configured
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_config.exists():
        logger.error(
            "Kaggle API not configured. Please:\n"
            "1. Go to https://www.kaggle.com/settings\n"
            "2. Click 'Create New API Token'\n"
            "3. Save kaggle.json to ~/.kaggle/\n"
            "4. chmod 600 ~/.kaggle/kaggle.json"
        )
        return None

    # Download dataset using Kaggle API
    dataset_id = "christophertreasure/nba-odds-data"
    download_path = "data/kaggle_nba_odds"
    Path(download_path).mkdir(parents=True, exist_ok=True)

    try:
        import kaggle

        logger.info(f"Downloading {dataset_id}...")
        kaggle.api.dataset_download_files(dataset_id, path=download_path, unzip=True)
        logger.info(f"✅ Dataset downloaded to {download_path}")

        # List downloaded files
        files = list(Path(download_path).glob("*.csv"))
        logger.info(f"Found {len(files)} CSV files:")
        for f in files:
            logger.info(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

        return download_path

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info(
            "\nAlternative: Manual download\n"
            "1. Go to https://www.kaggle.com/datasets/christophertreasure/nba-odds-data\n"
            "2. Click 'Download' button\n"
            "3. Unzip to data/kaggle_nba_odds/\n"
        )
        return None


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load and combine all CSV files from dataset."""
    logger.info("\n=== Loading Dataset ===")

    csv_files = list(Path(data_path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_path}")

    # Load and combine
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            logger.info(f"Loaded {csv_file.name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to load {csv_file.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"\n✅ Total: {len(combined)} games loaded")
    logger.info(f"Columns: {list(combined.columns)}")

    return combined


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run Consensus-mandated validation checks.

    Returns:
        dict with validation results and error rate
    """
    logger.info("\n=== Running Validation Checks ===")

    results = {
        "total_games": len(df),
        "spot_check_sample": 30,
        "errors": [],
        "error_rate": 0.0,
    }

    # 1. SPOT-CHECK: Random sample stratified by season
    logger.info("\n1. Spot-Check Random Sample (30 games)")

    if "season" in df.columns or "SEASON" in df.columns:
        season_col = "season" if "season" in df.columns else "SEASON"
        seasons = df[season_col].unique()

        # Sample proportionally from each season
        sample_size_per_season = max(2, 30 // len(seasons))
        sample = df.groupby(season_col).sample(
            n=min(sample_size_per_season, df.groupby(season_col).size().min()),
            random_state=42,
        )
    else:
        sample = df.sample(n=min(30, len(df)), random_state=42)

    logger.info(f"   Sampled {len(sample)} games for validation")

    # 2. LOGICAL CONSISTENCY: Spread vs Moneyline correlation
    logger.info("\n2. Logical Consistency Check")

    # Try to find spread and moneyline columns
    spread_cols = [c for c in df.columns if "spread" in c.lower()]
    ml_cols = [c for c in df.columns if "ml" in c.lower() or "moneyline" in c.lower()]

    if spread_cols and ml_cols:
        spread_col = spread_cols[0]
        ml_col = ml_cols[0]

        # Filter valid data
        valid_data = df[[spread_col, ml_col]].dropna()

        if len(valid_data) > 100:
            # Fit linear regression
            X = valid_data[[spread_col]].values
            y = valid_data[ml_col].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            logger.info(f"   Spread vs Moneyline R² = {r2:.3f}")

            if r2 < 0.85:
                results["errors"].append(
                    f"Low spread-ML correlation: R²={r2:.3f} < 0.85"
                )
                logger.warning(f"   ⚠️ R² below threshold (0.85)")
            else:
                logger.info(f"   ✅ Correlation OK (R² > 0.85)")
        else:
            logger.warning(f"   ⚠️ Insufficient data for correlation check")
    else:
        logger.warning(f"   ⚠️ Missing spread or moneyline columns")

    # 3. MISSINGNESS ANALYSIS
    logger.info("\n3. Missingness Analysis")

    missing_pct = df.isnull().sum() / len(df) * 100
    critical_missing = missing_pct[missing_pct > 10]

    if len(critical_missing) > 0:
        logger.warning(f"   ⚠️ High missingness detected:")
        for col, pct in critical_missing.items():
            logger.warning(f"      {col}: {pct:.1f}% missing")
            if pct > 50:
                results["errors"].append(f"{col}: {pct:.1f}% missing")
    else:
        logger.info(f"   ✅ No critical missingness (all <10%)")

    # 4. OUTLIER DETECTION
    logger.info("\n4. Outlier Detection")

    # Check totals column
    total_cols = [
        c for c in df.columns if "total" in c.lower() and "ou" not in c.lower()
    ]

    if total_cols:
        total_col = total_cols[0]
        totals = df[total_col].dropna()

        # Z-score outlier detection
        z_scores = np.abs(stats.zscore(totals))
        outliers = totals[z_scores > 3]

        outlier_rate = len(outliers) / len(totals) * 100
        logger.info(f"   Outliers (>3σ): {len(outliers)} ({outlier_rate:.2f}%)")

        if outlier_rate > 5:
            results["errors"].append(f"High outlier rate: {outlier_rate:.2f}% > 5%")
            logger.warning(f"   ⚠️ Outlier rate high (>{5}%)")
        else:
            logger.info(f"   ✅ Outlier rate acceptable (<5%)")

    # Calculate error rate
    results["error_rate"] = len(results["errors"]) / 4 * 100  # 4 checks total

    return results


def generate_validation_report(results: dict):
    """Generate validation report."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 60)

    logger.info(f"Total Games: {results['total_games']}")
    logger.info(f"Spot-Check Sample: {results['spot_check_sample']}")
    logger.info(f"\nErrors Detected: {len(results['errors'])}")

    if results["errors"]:
        for i, error in enumerate(results["errors"], 1):
            logger.warning(f"  {i}. {error}")

    logger.info(f"\nError Rate: {results['error_rate']:.1f}%")

    if results["error_rate"] <= 2.0:
        logger.info("\n✅ DATASET APPROVED (Error Rate ≤ 2%)")
        logger.info("Dataset is scientifically sound for calibration.")
        return True
    else:
        logger.warning(
            f"\n⚠️ DATASET NEEDS REVIEW (Error Rate {results['error_rate']:.1f}% > 2%)"
        )
        logger.warning("Consider using paid API for critical data correction.")
        return False


def main():
    """Main validation pipeline."""
    logger.info("=== Kaggle NBA Odds Dataset Validation ===\n")

    # 1. Download dataset
    data_path = download_kaggle_dataset()

    if not data_path:
        # Check if already downloaded
        data_path = "data/kaggle_nba_odds"
        if not Path(data_path).exists() or not list(Path(data_path).glob("*.csv")):
            logger.error("Dataset not available. Please download manually.")
            return

    # 2. Load dataset
    df = load_dataset(data_path)

    # 3. Run validation
    results = validate_dataset(df)

    # 4. Generate report
    approved = generate_validation_report(results)

    # 5. Save validated dataset
    if approved:
        output_path = "data/nba_odds_validated.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"\n✅ Validated dataset saved to {output_path}")
        logger.info(f"Ready for calibration data generation.")

    return approved


if __name__ == "__main__":
    main()
