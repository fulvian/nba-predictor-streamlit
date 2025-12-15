# Recent Changes (v2.2.0)

## Multi-Persona NBA Consensus System

### Architecture Changes
- **4-Agent Prompt System**: Replaced single "Sharp NBA Bettor" persona with 4 specialized agents:
  1. **The Quant**: Pure statistical analysis, implied probability calculation
  2. **The Scout**: Narrative analysis, qualitative factors, epistemic uncertainty
  3. **The Bookmaker**: Risk calibration, EV calculation, aleatoric uncertainty
  4. **The Moderator**: Bayesian fusion with weighted synthesis (40%/30%/30%)
- **Chain-of-Thought Reasoning**: Each persona explicitly reasons step-by-step before concluding
- **Uncertainty Decomposition**: Separates aleatoric (inherent variance) from epistemic (missing info)

### New Output Fields
- `persona_votes`: Individual adjustments from each persona (quant, scout, bookmaker)
- `ev_assessment`: Expected Value classification (POSITIVE/NEGATIVE/NEUTRAL)

### Configuration Updates
- **max_tokens**: Increased from 2048 → 16384 in `NanoGPT-Consensus-MCP/client.py`
- **timeout**: Increased from 180s → 600s in `nanogpt_client.py`

## Bug Fixes
- **402 Payment Required**: Resolved by removing incompatible memory headers for Flat Plan
- **Token Limit Hit**: Thinking models now have adequate tokens for full reasoning

## Files Modified
- `src/nba_predictor/intelligence/nanogpt_client.py`: Multi-persona prompt + 600s timeout
- `NanoGPT-Consensus-MCP/src/nanogpt_consensus/client.py`: 16384 max_tokens
- `NanoGPT-Consensus-MCP/src/nanogpt_consensus/consensus.py`: Disabled memory headers

## Verification
- Tested with Lakers vs Celtics sample: 3/3 models returned valid multi-persona JSON
- DeepSeek R1, Kimi K2, Qwen 3 all producing `persona_votes` and `ev_assessment`
