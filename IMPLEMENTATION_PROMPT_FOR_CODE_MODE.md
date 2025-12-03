# NBA Predictor Refactoring - Implementation Prompt

## 🎯 Task Description

Implementa il piano di refactoring del sistema NBA Predictor basato sull'analisi architetturale completa. Il sistema presenta problemi di degradazione dovuti a evoluzione iterativa non gestita, con componenti legacy affiancati a implementazioni moderne.

## 📋 Context Summary

### Current State Problems
1. **Pipeline Proliferation**: 3 pipeline di prediction sovrapposte
2. **Component Duplication**: Componenti duplicati che svolgono le stesse funzioni
3. **Legacy Code Bloat**: 20+ script deprecated nel sistema
4. **Data Fragmentation**: Dati sparsi su formati e location multiple

### Target Architecture
- Single ML pipeline: `UnifiedHybridPipeline`
- Unified data store: `UnifiedDataStore`
- Consolidated ML interface: `NBAEnsemblePredictor`
- Clean Streamlit dashboard without duplicates

## 🚀 Implementation Tasks

### Phase 1: Immediate Cleanup (Week 1)

#### Task 1.1: Remove Deprecated Components
**Objective**: Remove all legacy code that's no longer used

**Files to Remove**:
```bash
# Remove entire deprecated directory
rm -rf deprecated/

# Remove legacy pipelines
rm src/nba_predictor/core/enhanced_prediction_pipeline.py
rm src/nba_predictor/core/prediction_pipeline.py

# Remove duplicate bridges
rm src/nba_predictor/streamlit/components/enhanced_prediction_bridge.py
```

**Validation**: Ensure system still works after removal

#### Task 1.2: Update All Imports
**Objective**: Replace all imports of removed components with production ones

**Search and Replace Pattern**:
```python
# OLD (remove these imports)
from nba_predictor.core.enhanced_prediction_pipeline import EnhancedPredictionPipeline
from nba_predictor.core.prediction_pipeline import PredictionPipeline
from nba_predictor.streamlit.components.enhanced_prediction_bridge import EnhancedPredictionBridge

# NEW (use these everywhere)
from nba_predictor.core.unified_hybrid_pipeline import UnifiedHybridPipeline
from nba_predictor.streamlit.components.enhanced_prediction_bridge_v2 import EnhancedPredictionBridgeV2
```

**Files to Update**:
- `src/nba_predictor/streamlit/app.py`
- `src/nba_predictor/streamlit/betting_workflow_dashboard.py`
- `src/nba_predictor/streamlit/components/ml_integration_bridge.py`
- `src/nba_predictor/ensemble/nba_ensemble_predictor.py`
- `src/nba_predictor/integration/modern_prediction_system.py`

#### Task 1.3: Clean Up Legacy Dashboard Components
**Objective**: Remove unused dashboard components and consolidate functionality

**Actions**:
1. Identify unused dashboard files
2. Remove duplicate functionality
3. Standardize on main dashboard: `src/nba_predictor/streamlit/betting_workflow_dashboard.py`

### Phase 2: Pipeline Consolidation (Week 2)

#### Task 2.1: Enforce UnifiedHybridPipeline Everywhere
**Objective**: Standardize all components to use UnifiedHybridPipeline

**Implementation**:
```python
# Create unified ML interface
class UnifiedMLInterface:
    def __init__(self):
        self.pipeline = UnifiedHybridPipeline()
        self.ensemble = NBAEnsemblePredictor()
    
    def predict(self, team1, team2, line, home_team=None):
        """Unified prediction interface"""
        return self.pipeline.predict_unified(team1, team2, line, home_team)
```

**Files to Create**:
- `src/nba_predictor/core/unified_ml_interface.py`

#### Task 2.2: Update ML Integration Bridge
**Objective**: Simplify ML integration by removing duplicate ensemble predictors

**Actions**:
1. Remove duplicate ensemble logic
2. Standardize on NBAEnsemblePredictor
3. Implement single prediction interface
4. Update all calls to use unified interface

#### Task 2.3: Consolidate Model Management
**Objective**: Keep only production models and remove duplicates

**Models to Keep**:
- LightGBM (NBA optimized): `src/nba_predictor/models/lightgbm_model.py`
- XGBoost + Neural Network ensemble: `src/nba_predictor/ensemble/nba_ensemble_predictor.py`
- Stacked ensemble (if needed): `src/nba_predictor/models/stacked_ensemble.py`

**Models to Remove/Consolidate**:
- Duplicate models in `src/models/nba_models.py`
- Experimental models not used in production

### Phase 3: Data Layer Unification (Week 3)

#### Task 3.1: Unified Data Store Standardization
**Objective**: Enforce UnifiedDataStore everywhere for data access

**Implementation Pattern**:
```python
# Standard data access pattern
from nba_predictor.core.data_store import UnifiedDataStore

class DataManager:
    def __init__(self):
        self.data_store = UnifiedDataStore()
    
    def get_game_data(self, game_id):
        """Get game data through unified interface"""
        return self.data_store.get_game_data(game_id)
```

**Files to Update**:
- All files that directly access DuckDB or Parquet files
- API providers to use UnifiedDataStore for caching
- ML pipelines to use UnifiedDataStore for training data

#### Task 3.2: Database Cleanup
**Objective**: Consolidate DuckDB databases and implement proper backup

**Actions**:
1. Identify primary database: `data/nba_betting.duckdb`
2. Remove backup databases after verification
3. Implement proper backup strategy
4. Update all database connections to use primary database

#### Task 3.3: Cache Management Implementation
**Objective**: Implement intelligent caching system

**Files to Create**:
- `src/nba_predictor/utils/cache_manager.py`

**Implementation**:
```python
class CacheManager:
    def __init__(self, cache_dir=".nba_cache/"):
        self.cache_dir = cache_dir
    
    def get_cached_prediction(self, key):
        """Get cached prediction if available"""
        pass
    
    def cache_prediction(self, key, prediction, ttl=3600):
        """Cache prediction with TTL"""
        pass
```

### Phase 4: Feature Engineering Optimization (Week 4)

#### Task 4.1: Feature Validation and Optimization
**Objective**: Analyze and optimize the current 25+ features

**Current Features to Validate**:
- **Four Factors Features** (8): eFG%, TOV%, OREB%, FT_RATE%
- **Team Differentials** (10): Score, rebounds, assists differentials
- **Pace Features** (5): HOME_PACE, AWAY_PACE, GAME_PACE
- **Efficiency Features** (4): Offensive/Defensive ratings
- **Situational Features** (6): Home/away, rest days, schedule

**Actions**:
1. Analyze feature importance using SHAP values
2. Remove redundant features (high correlation)
3. Optimize feature computation performance
4. Validate feature ranges for NBA realism

#### Task 4.2: Feature Engineering Pipeline
**Objective**: Standardize feature creation in single pipeline

**Files to Create**:
- `src/nba_predictor/features/unified_feature_pipeline.py`

**Implementation**:
```python
class UnifiedFeaturePipeline:
    def __init__(self):
        self.feature_engineers = {
            'four_factors': FourFactorsEngineer(),
            'team_differentials': TeamDifferentialEngineer(),
            'pace_features': PaceFeatureEngineer(),
            'efficiency_features': EfficiencyEngineer(),
            'situational_features': SituationalFeatureEngineer()
        }
    
    def extract_features(self, game_data):
        """Extract all features in unified pipeline"""
        features = {}
        for name, engineer in self.feature_engineers.items():
            features.update(engineer.extract_features(game_data))
        return features
```

#### Task 4.3: Model Performance Optimization
**Objective**: Optimize ensemble performance to maintain < 20ms prediction time

**Actions**:
1. Implement model caching
2. Optimize ensemble weights
3. Add performance monitoring
4. Implement prediction batching for efficiency

## 🔧 Technical Requirements

### Code Quality Standards
- Maintain > 95% test coverage
- Follow PEP 8 for Python code
- Add comprehensive docstrings
- Implement proper error handling
- Use type hints throughout

### Performance Requirements
- Prediction latency: < 20ms
- Memory usage: reduce by 25%
- Startup time: improve by 40%
- Cache hit rate: > 80%

### Testing Requirements
- Unit tests for all new components
- Integration tests for pipeline changes
- Performance benchmarks
- Regression tests for existing functionality

## 📊 Success Metrics

### Technical Metrics
- Code reduction: 30% reduction in lines of code
- File count: Reduce from 500+ to ~350 files
- Import complexity: Reduce circular dependencies by 50%

### Performance Metrics
- Memory usage: Reduce by 25%
- Startup time: Improve by 40%
- Cache hit rate: Achieve > 80%
- Test coverage: Maintain > 95%

### Business Metrics
- Development velocity: Improve feature delivery time by 50%
- Bug reduction: Reduce bug count by 60%
- Maintenance overhead: Reduce maintenance time by 40%

## 🚨 Risk Mitigation

### Backup Strategy
1. Create full backup before each phase
2. Use git branches for each phase
3. Implement rollback capability
4. Test thoroughly before merging

### Testing Strategy
1. Comprehensive test suite before each phase
2. Performance benchmarks after changes
3. Integration tests for pipeline changes
4. User acceptance testing for UI changes

### Deployment Strategy
1. Implement changes incrementally
2. Test in development environment first
3. Staging environment validation
4. Production deployment with monitoring

## 📅 Implementation Order

### Week 1: Cleanup
1. Remove deprecated components
2. Update all imports
3. Clean up dashboard components

### Week 2: Pipeline Consolidation
1. Enforce UnifiedHybridPipeline
2. Update ML integration bridge
3. Consolidate model management

### Week 3: Data Unification
1. Unified data store standardization
2. Database cleanup
3. Cache management implementation

### Week 4: Feature Optimization
1. Feature validation and optimization
2. Feature engineering pipeline
3. Model performance optimization

## 🎯 Expected Outcomes

### Immediate Benefits
- Cleaner codebase with 30% less code
- Reduced confusion from duplicate components
- Faster build and startup times
- Improved developer experience

### Medium-term Benefits
- Consistent architecture across all components
- Better performance and reliability
- Easier maintenance and debugging
- Improved scalability

### Long-term Benefits
- Sustainable development workflow
- Faster feature development
- Better code quality
- Easier onboarding for new developers

## 🔄 Validation Steps

### After Each Phase
1. Run full test suite
2. Verify performance benchmarks
3. Check functionality integrity
4. Validate user experience

### Final Validation
1. Complete system integration test
2. Performance benchmark validation
3. User acceptance testing
4. Documentation completeness check

---

**Important**: Implement this refactoring plan incrementally, validating each phase before proceeding to the next. Focus on maintaining system functionality while improving code quality and performance.

**Reference Documents**:
- `NBA_PREDICTOR_ARCHITECTURE_ANALYSIS.md` - Detailed system analysis
- `NBA_PREDICTOR_REFACTORING_PLAN.md` - Complete refactoring strategy

**Start with Phase 1 and proceed sequentially through all phases.**