    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save trained model and metadata to disk.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(self.model_path / f"unified_model_{timestamp}.pkl")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_package = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "scaler": getattr(self, "scaler", None),
            "training_date": datetime.now().isoformat(),
            "training_samples": len(self.feature_columns) if hasattr(self, "feature_columns") else 0,
            "model_version": "1.0_rolling5",
            "use_stacked_ensemble": self.use_stacked_ensemble,
        }
        
        with open(filepath, 'wb') as f:
            joblib.dump(model_package, f)
        
        logger.info(f"✅ Model saved to {filepath}")
        
        # Also save as 'latest' for easy loading
        latest_path = str(self.model_path / "unified_model_latest.pkl")
        with open(latest_path, 'wb') as f:
            joblib.dump(model_package, f)
        logger.info(f"✅ Model also saved as latest: {latest_path}")
        
        return filepath

    def load_model(self, filepath: Optional[str] = None) -> bool:
        """
        Load model from disk with validation.
        
        Args:
            filepath: Path to model file (defaults to latest)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if filepath is None:
            filepath = str(self.model_path / "unified_model_latest.pkl")
            
        if not os.path.exists(filepath):
            logger.warning(f"⚠️ Model file not found: {filepath}")
            return False
            
        try:
            with open(filepath, 'rb') as f:
                model_package = joblib.load(f)
            
            # Validate model package
            required_keys = ["model", "feature_columns", "model_version"]
            if not all(key in model_package for key in required_keys):
                logger.warning("⚠️ Model package incomplete, will retrain")
                return False
            
            # Restore state
            self.model = model_package["model"]
            self.feature_columns = model_package["feature_columns"]
            if "scaler" in model_package:
                self.scaler = model_package["scaler"]
            self.is_trained = True
            
            logger.info(f"✅ Model loaded from {filepath}")
            logger.info(f"   Version: {model_package.get('model_version', 'unknown')}")
            logger.info(f"   Training date: {model_package.get('training_date', 'unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

    def should_retrain(self, force: bool = False) -> bool:
        """
        Check if model needs retraining based on data staleness.
        
        Args:
            force: Force retraining regardless of conditions
            
        Returns:
            True if retraining is needed
        """
        if force:
            logger.info("🔄 Force retraining requested")
            return True
            
        if not self.is_trained:
            logger.info("🔄 Model not trained, training needed")
            return True
        
        # Check if new games available
        try:
            current_games_count = len(self.data_store.load_nba_real_games())
            
            # Load training metadata from latest model
            latest_model_path = str(self.model_path / "unified_model_latest.pkl")
            if os.path.exists(latest_model_path):
                with open(latest_model_path, 'rb') as f:
                    model_package = joblib.load(f)
                    
                training_samples = model_package.get("training_samples", 0)
                new_games = current_games_count - training_samples
                
                # Retrain if > 50 new games (approximately 1 week of NBA games)
                if new_games > 50:
                    logger.info(f"🔄 {new_games} new games detected (threshold: 50), retraining needed")
                    return True
                else:
                    logger.info(f"✅ Only {new_games} new games, using cached model")
                    return False
            else:
                logger.info("🔄 No cached model found, training needed")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ Error checking staleness: {e}, will retrain")
            return True
