# ADR-004: Multi-Persona NBA Consensus System

## Status

**Accepted** - 2025-12-15

## Context

The original NBA Consensus prompt used a single "Sharp NBA Bettor" persona. Research (Dec 2025) showed that multi-persona dialogue systems outperform single-expert approaches by 15-20% in uncertainty management for sports betting.

### Problems with Single Persona

1. **Confirmation Bias**: Single expert tends to reinforce initial assumptions
2. **No Uncertainty Decomposition**: Mixed aleatoric and epistemic uncertainty
3. **Missing EV Calculation**: No explicit Expected Value assessment
4. **Black Box Reasoning**: Difficult to debug when predictions fail

## Decision

Implement a **4-Agent Multi-Persona System** with Chain-of-Thought reasoning:

### Agent Architecture

| Agent | Role | Weight | Key Outputs |
|-------|------|--------|-------------|
| **The Quant** | Statistical Analyst | 40% | `quant_adjustment`, `quant_confidence` |
| **The Scout** | Narrative Analyst | 30% | `scout_adjustment`, `epistemic_uncertainty` |
| **The Bookmaker** | Risk Manager | 30% | `bookmaker_adjustment`, `aleatoric_uncertainty`, `ev_assessment` |
| **The Moderator** | Synthesizer | - | Final weighted consensus |

### Prompt Flow

```
Input Data → Quant (baseline) → Scout (challenge) → Bookmaker (risk) → Moderator (fusion) → JSON
```

### New Output Schema

```json
{
  "point_adjustment": 3.4,
  "uncertainty_factor": 0.38,
  "risk_level": "MED",
  "reasoning": "Multi-persona synthesis...",
  "persona_votes": {
    "quant": 5.0,
    "scout": -0.5,
    "bookmaker": 4.0
  },
  "ev_assessment": "POSITIVE"
}
```

## Configuration Changes

- **max_tokens**: 2048 → 16384 (thinking models need headroom for CoT)
- **timeout**: 180s → 600s (10 min for 3 parallel thinking models)
- **Memory headers**: Disabled for Flat Plan compatibility

## Consequences

### Positive

- **Transparency**: `persona_votes` shows disagreement between agents
- **Debuggability**: Can trace prediction errors to specific persona
- **Better Calibration**: Explicit uncertainty decomposition
- **EV Awareness**: Explicit assessment of betting edge

### Negative

- **Longer Response Time**: ~2-4 minutes vs ~30s for old prompt
- **Higher Token Usage**: ~5000-15000 tokens per model vs ~2000

### Neutral

- **Backward Compatible**: Old fields (`point_adjustment`, `uncertainty_factor`, `risk_level`, `reasoning`) still present

## Files Modified

- `src/nba_predictor/intelligence/nanogpt_client.py`: `_construct_prompt()`, `_aggregate_consensus_results()`
- `src/nba_predictor/core/unified_hybrid_pipeline.py`: timeout parameter
- `NanoGPT-Consensus-MCP/src/nanogpt_consensus/client.py`: max_tokens
- `NanoGPT-Consensus-MCP/src/nanogpt_consensus/consensus.py`: memory headers

## Verification

Tested with sample matchups (Lakers vs Celtics, Celtics vs Pistons, Jazz vs Mavericks):
- 3/3 models (DeepSeek R1, Kimi K2, Qwen 3) return valid multi-persona JSON
- `persona_votes` correctly populated with distinct values per agent
- `ev_assessment` provides actionable signal
