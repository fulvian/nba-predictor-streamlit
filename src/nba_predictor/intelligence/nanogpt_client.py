import asyncio
import json
import logging
import os
import sys
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


class SimpleMCPClient:
    """
    A lightweight, dependency-free MCP Client implementation for Python 3.9+.
    Handles JSON-RPC 2.0 over Stdio.
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd
        self.process = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}

    async def start(self):
        """Start the subprocess."""
        logger.info(
            f"🚀 Starting NanoGPT process: {self.command} {self.args} (cwd={self.cwd})"
        )
        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
            cwd=self.cwd,
        )
        # Start reading loop
        asyncio.create_task(self._read_stdout())
        asyncio.create_task(self._read_stderr())

        # Initialize MCP handshake
        await self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",  # Current MCP version
                "capabilities": {},
                "clientInfo": {"name": "antigravity-client", "version": "1.0"},
            },
        )
        # Notify initialized
        await self.send_notification("notifications/initialized", {})

    async def stop(self):
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except Exception as e:
                logger.error(f"Error stopping process: {e}")

    async def _read_stdout(self):
        """Read standard output line by line (JSON-RPC messages)."""
        while True:
            try:
                # logger.info("DEBUG: Background readline...")
                line = await self.process.stdout.readline()
                logger.info(f"DEBUG RX: {line}")
                if not line:
                    break
                line_str = line.decode().strip()
                if not line_str:
                    continue

                try:
                    message = json.loads(line_str)
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON stdout: {line_str}")
            except Exception as e:
                logger.error(f"Error reading stdout: {e}")
                break

    async def _read_stderr(self):
        """Read stderr for logging."""
        while True:
            line = await self.process.stderr.readline()
            if not line:
                break
            logger.warning(f"NanoGPT Stderr: {line.decode().strip()}")

    async def _receive(self) -> Optional[dict]:
        """Read a JSON-RPC message line."""
        if self.process.stdout:
            # logger.info("DEBUG: Awaiting readline...")
            line = await self.process.stdout.readline()
            logger.info(f"DEBUG RAW LINE: {line}")
            if line:
                logger.info(f"DEBUG CLIENT RECV: {line.decode().strip()}")
                return json.loads(line.decode().strip())
        return None

    async def _handle_message(self, message: dict):
        """Handle incoming JSON-RPC message."""
        if "id" in message and "result" in message:
            # Response to request
            req_id = message["id"]
            if req_id in self._pending_requests:
                self._pending_requests[req_id].set_result(message["result"])
                del self._pending_requests[req_id]
        elif "id" in message and "error" in message:
            # Error response
            req_id = message["id"]
            if req_id in self._pending_requests:
                self._pending_requests[req_id].set_exception(
                    RuntimeError(message["error"])
                )
                del self._pending_requests[req_id]

    async def send_request(self, method: str, params: Optional[dict] = None) -> Any:
        self._request_id += 1
        req_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": req_id,
        }

        future = asyncio.Future()
        self._pending_requests[req_id] = future

        input_data = json.dumps(request) + "\n"
        self.process.stdin.write(input_data.encode())
        await self.process.stdin.drain()

        return await future

    async def send_notification(self, method: str, params: Optional[dict] = None):
        request = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        input_data = json.dumps(request) + "\n"
        # Check if stdin is closed before writing
        if self.process.stdin.is_closing():
            logger.warning("Attempted to write to closed stdin")
            return
        self.process.stdin.write(input_data.encode())
        await self.process.stdin.drain()

    async def call_tool(self, name: str, arguments: dict) -> Any:
        return await self.send_request(
            "tools/call", {"name": name, "arguments": arguments}
        )


class NanoGPTClient:
    """
    Client for the NanoGPT Consensus Engine via custom MCP Protocol (StdIO).
    Acts as the bridge for the "Reasoning Expert".
    """

    def __init__(self, timeout: int = 600):
        self.timeout = timeout

        # Load config
        self.command = os.getenv("NANOGPT_CONSENSUS_COMMAND", "uvx")
        # Updated to use uv run for direct execution of local code (bypasses cache)
        args_str = os.getenv(
            "NANOGPT_CONSENSUS_ARGS",
            '["uv", "run", "--directory", "/Users/fulvioventura/NanoGPT-Consensus-MCP", "python", "-m", "nanogpt_consensus.server"]',
        )
        self.cwd = os.getenv("NANOGPT_CONSENSUS_CWD", None)
        try:
            self.args = json.loads(args_str)
        except Exception:
            self.args = [
                "uv",
                "run",
                "--directory",
                "/Users/fulvioventura/NanoGPT-Consensus-MCP",
                "python",
                "-m",
                "nanogpt_consensus.server",
            ]

    async def _query_consensus_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        complexity: str = "nba_predictor",
    ) -> dict[str, Any]:
        """
        Connects to the server, runs the query, and disconnects.
        (For production, we might want a persistent connection, but per-request is safer for stability).
        """
        client = SimpleMCPClient(
            self.command, self.args, env=os.environ.copy(), cwd=self.cwd
        )

        try:
            await client.start()

            tool_args = {"prompt": prompt, "complexity": complexity}
            if system_prompt:
                tool_args["system_prompt"] = system_prompt

            logger.info(f"🧠 Querying Consensus: {prompt[:50]}...")

            # The tool result structure from MCP is specific:
            # { content: [ { type: "text", text: "..." } ], isError: false }
            result = await asyncio.wait_for(
                client.call_tool("ask_consensus", tool_args), timeout=self.timeout
            )

            final_text = ""
            if "content" in result and isinstance(result["content"], list):
                for item in result["content"]:
                    if item.get("type") == "text":
                        final_text += item.get("text", "")

            try:
                return json.loads(final_text)
            except json.JSONDecodeError:
                return {"raw_response": final_text}

        except Exception as e:
            logger.error(f"❌ NanoGPT Consensus Call Failed: {repr(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return {"error": str(e), "fallback": True}
        finally:
            await client.stop()

    def query_consensus_sync(
        self,
        context: dict[str, Any],
        complexity: str = "nba_predictor",
        meta_learning_context: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Synchronous wrapper for integration into the blocking pipeline.

        Args:
            context: Dictionary containing stats, news, and match info.
            complexity: Complexity level/profile for consensus (default: "nba_predictor").
            meta_learning_context: Optional string containing feedback loop insights.
        """

        # Construct the Prompt (System 2 Thinking)
        prompt = self._construct_prompt(context, meta_learning_context)
        system_prompt = "You are a multi-agent NBA consensus system. Execute all 4 analysis phases (Quant, Scout, Bookmaker, Moderator) sequentially, then output the final JSON synthesis."

        try:
            # Event Loop Safety for Streamlit
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # We are inside an active event loop (e.g., Streamlit)
                # We must use it rather than creating a new one with asyncio.run()
                # However, calling run_until_complete() on a running loop is blocked.
                # We need to run this in a separate thread or use nest_asyncio.
                # For simplicity/robustness without extra deps, we create a new loop in a thread if needed,
                # BUT since this is a heavy blocking call, just running it standard asyncio.run might fail.

                # TRICK: Apply nest_asyncio if available, otherwise warn.
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                    raw_response = asyncio.run(
                        self._query_consensus_async(prompt, system_prompt, complexity)
                    )
                except ImportError:
                    logger.warning(
                        "⚠️ nest_asyncio not found. Attempting risky loop interaction."
                    )
                    # Fallback: Just return a task? No, this function must be sync.
                    # We can try creating a new event loop policy?
                    # Let's try just running it and catch RuntimeError
                    raw_response = asyncio.run(
                        self._query_consensus_async(prompt, system_prompt, complexity)
                    )
            else:
                raw_response = asyncio.run(
                    self._query_consensus_async(prompt, system_prompt, complexity)
                )

            return self._aggregate_consensus_results(raw_response)
        except Exception as e:
            logger.error(f"❌ NanoGPT Consensus Sync Error: {e}")
            import traceback

            tb = traceback.format_exc()
            logger.error(tb)
            return {
                "error": type(e).__name__,
                "details": str(e) or "Check logs for traceback",
                "fallback": True,
            }

    def _aggregate_consensus_results(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate results from multiple models in the consensus response.
        Updated for Hybrid Strategy: Extracts both Point Adjustment and Uncertainty Factor.
        """
        try:
            results = response.get("results", [])
            if not results:
                # Check if it's a single response format (fallback)
                if "point_adjustment" in response:
                    return response
                return {"error": "No results found", "fallback": True}

            total_adj = 0.0
            total_uncertainty = 0.0
            valid_count = 0
            risk_levels = []
            reasonings = []
            all_persona_votes = []
            all_ev_assessments = []

            for res in results:
                try:
                    content_str = res.get("content") or "{}"
                    # Extract JSON from markdown code block if present
                    if "```json" in content_str:
                        content_str = (
                            content_str.split("```json")[1].split("```")[0].strip()
                        )
                    elif "```" in content_str:
                        content_str = content_str.split("```")[1].strip()

                    data = json.loads(content_str)

                    # Extract Bias (Point Adjustment)
                    total_adj += float(data.get("point_adjustment", 0.0))

                    # Extract Uncertainty/Volatility
                    # Some models might optionally return 'confidence' instead of uncertainty.
                    # We map: Uncertainty = 1 - (Confidence/100) or use explicit 'uncertainty_factor'
                    unc = data.get("uncertainty_factor")
                    if unc is None:
                        conf = float(data.get("confidence", 50.0))
                        unc = 1.0 - (conf / 100.0)
                    total_uncertainty += float(unc)

                    risk_levels.append(data.get("risk_level", "HIGH").upper())
                    reasonings.append(
                        f"{res.get('model', 'Model')}: {data.get('reasoning', '')}"
                    )

                    # NEW: Extract multi-persona fields if present
                    if "persona_votes" in data:
                        all_persona_votes.append(data["persona_votes"])
                    if "ev_assessment" in data:
                        all_ev_assessments.append(data["ev_assessment"])

                    valid_count += 1
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse model result: {e}")
                    continue

            if valid_count == 0:
                return {"error": "No valid model results", "fallback": True}

            avg_adj = total_adj / valid_count
            avg_uncertainty = total_uncertainty / valid_count

            # Determine aggregate risk (Conservatively take the highest reported risk)
            if "HIGH" in risk_levels:
                final_risk = "HIGH"
            elif "MED" in risk_levels:
                final_risk = "MED"
            else:
                final_risk = "LOW"

            # Aggregate EV assessments (majority vote)
            final_ev = "NEUTRAL"
            if all_ev_assessments:
                from collections import Counter

                ev_counts = Counter(all_ev_assessments)
                final_ev = ev_counts.most_common(1)[0][0]

            # Aggregate persona votes (average across models)
            final_persona_votes = None
            if all_persona_votes:
                final_persona_votes = {
                    "quant": sum(pv.get("quant", 0) for pv in all_persona_votes)
                    / len(all_persona_votes),
                    "scout": sum(pv.get("scout", 0) for pv in all_persona_votes)
                    / len(all_persona_votes),
                    "bookmaker": sum(pv.get("bookmaker", 0) for pv in all_persona_votes)
                    / len(all_persona_votes),
                }

            result = {
                "point_adjustment": avg_adj,
                "uncertainty_factor": avg_uncertainty,
                "risk_level": final_risk,
                "reasoning": " | ".join(reasonings[:3]),  # Limit to 3 for brevity
                "model_count": valid_count,
                "ev_assessment": final_ev,
            }

            # Add persona votes if available
            if final_persona_votes:
                result["persona_votes"] = final_persona_votes

            return result

        except Exception as e:
            logger.error(f"Failed to aggregate consensus results: {e}")
            return {"error": str(e), "fallback": True}

    def _construct_prompt(
        self, context: dict[str, Any], meta_learning_context: Optional[str] = None
    ) -> str:
        """
        Multi-Persona Consensus Prompt System (v2.0)

        Architecture: 4 specialized agents debate sequentially:
        1. The Quant - Statistical baseline + confidence interval
        2. The Scout - Narrative analysis + qualitative adjustments
        3. The Bookmaker - Risk calibration + EV calculation
        4. The Moderator - Bayesian fusion of all perspectives

        Based on Dec 2025 research: Multi-persona outperforms single expert 15-20%
        """
        team1 = context.get("team1", "Team A")
        team2 = context.get("team2", "Team B")
        predicted_total = context.get("predicted_total", "N/A")
        market_line = context.get("market_line", "N/A")
        deviation = context.get("deviation_from_market", "N/A")
        confidence = context.get("confidence", "N/A")
        stats = context.get("stats", {})

        # Build news context
        news = context.get("news", [])
        news_text = ""
        if news:
            news_items = [
                f"- [{item.get('type', 'news').upper()}]: {item.get('text', '')}"
                for item in news[:10]
            ]
            news_text = "\n".join(news_items)
        else:
            news_text = "- No significant recent news found."

        prompt = f"""# Multi-Agent NBA Consensus Analysis
**Matchup**: {team1} vs {team2}

You are a multi-agent consensus system. Analyze this game through 4 sequential expert perspectives, then synthesize a final prediction.

---

## 📊 INPUT DATA

### Quantitative Baseline
- **Model Prediction**: {predicted_total} pts
- **Market Line**: {market_line}
- **Deviation from Market**: {deviation}
- **Model Confidence**: {confidence}
- **Stats**: {json.dumps(stats, indent=2)}

### News & Context
{news_text}

"""
        # Insert Meta-Learning Context if available
        if meta_learning_context:
            prompt += f"""### Historical Bias Correction (Meta-Learning)
{meta_learning_context}

"""

        prompt += """---

## 🧠 PHASE 1: THE QUANT (Statistical Analyst)

*Role*: Pure data-driven analysis. Ignore narratives.

**Chain-of-Thought**:
1. Calculate implied probability from market odds (e.g., O/U at -110 → 52.4% each side)
2. Estimate true probability from model deviation
3. Identify statistical edge (if any)
4. Propose baseline adjustment from PURE STATISTICS

**Output** (think step-by-step, then conclude):
- `quant_adjustment`: (float) Statistical-only adjustment
- `quant_confidence`: (float 0-1) Statistical confidence

---

## 🔍 PHASE 2: THE SCOUT (Narrative Analyst)

*Role*: Qualitative factors statistics MISS. Challenge The Quant.

**Chain-of-Thought**:
1. Identify key injuries/absences and their REAL impact (beyond box score)
2. Assess coaching strategy, rest patterns, motivation factors
3. Evaluate if news is ALREADY priced in by market
4. Propose narrative-driven adjustment to baseline

**Input**: Quant's baseline adjustment (from Phase 1)
**Output** (think step-by-step, then conclude):
- `scout_adjustment`: (float) Narrative-only adjustment
- `epistemic_uncertainty`: (float 0-1) Missing information uncertainty

---

## 💰 PHASE 3: THE BOOKMAKER (Risk Manager)

*Role*: Market efficiency + uncertainty decomposition.

**Chain-of-Thought**:
1. Is the market line efficient? Check deviation significance.
2. Decompose uncertainty:
   - **Aleatoric** (inherent variance): shooting variance, referee calls, random events
   - **Epistemic** (missing info): unknown player status, locker room issues
3. Calculate Expected Value: EV = (true_prob × payout) - (false_prob × stake)
4. Assess if edge is worth the risk

**Input**: Quant + Scout adjustments (from Phases 1-2)
**Output** (think step-by-step, then conclude):
- `bookmaker_adjustment`: (float) Risk-adjusted final adjustment
- `aleatoric_uncertainty`: (float 0-1) Inherent game variance
- `ev_assessment`: "POSITIVE" | "NEGATIVE" | "NEUTRAL"

---

## ⚖️ PHASE 4: THE MODERATOR (Strategic Integrator)

*Role*: Bayesian fusion of all 3 perspectives.

**Weighting Schema**:
- Quant: 40% (statistical floor)
- Scout: 30% (narrative adjustment)
- Bookmaker: 30% (risk calibration)

**Chain-of-Thought**:
1. Review all 3 agent outputs
2. Identify agreement vs disagreement
3. Weight by confidence levels
4. Synthesize final consensus

---

## 🎯 FINAL OUTPUT (JSON ONLY)

After completing all 4 phases, output ONLY this JSON:

```json
{
  "point_adjustment": <weighted consensus adjustment, float, max +/-15>,
  "uncertainty_factor": <combined uncertainty 0.0-1.0, where 0=stable, 1=chaos>,
  "risk_level": "<LOW|MED|HIGH based on epistemic uncertainty>",
  "reasoning": "<2-3 sentence synthesis of all perspectives>",
  "persona_votes": {
    "quant": <quant's adjustment>,
    "scout": <scout's adjustment>,
    "bookmaker": <bookmaker's adjustment>
  },
  "ev_assessment": "<POSITIVE|NEGATIVE|NEUTRAL>"
}
```

**IMPORTANT**: 
- Complete ALL 4 phases before outputting JSON
- The `reasoning` must reference insights from MULTIPLE personas
- If personas DISAGREE significantly, increase `uncertainty_factor`
- Output ONLY the final JSON, no markdown code blocks
"""
        return prompt
