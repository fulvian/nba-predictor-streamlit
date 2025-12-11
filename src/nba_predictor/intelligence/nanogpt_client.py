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

    def __init__(self, timeout: int = 180):
        self.timeout = timeout

        # Load config
        self.command = os.getenv("NANOGPT_CONSENSUS_COMMAND", "uvx")
        args_str = os.getenv("NANOGPT_CONSENSUS_ARGS", '["nanogpt-consensus"]')
        self.cwd = os.getenv("NANOGPT_CONSENSUS_CWD", None)
        try:
            self.args = json.loads(args_str)
        except Exception:
            self.args = ["nanogpt-consensus"]

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
        self, context: dict[str, Any], complexity: str = "nba_predictor"
    ) -> dict[str, Any]:
        """
        Synchronous wrapper for integration into the blocking pipeline.

        Args:
            context: Dictionary containing stats, news, and match info.
        """

        # Construct the Prompt (System 2 Thinking)
        prompt = self._construct_prompt(context)
        system_prompt = "You are a specialized NBA Reasoning Expert. Analyze the quantitative data and qualitative news to provide a consensus prediction."

        try:
            raw_response = asyncio.run(
                self._query_consensus_async(prompt, system_prompt, complexity)
            )
            return self._aggregate_consensus_results(raw_response)
        except Exception as e:
            logger.error(f"❌ NanoGPT Consensus Sync Error: {e}")
            return {"error": str(e), "fallback": True}

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

            for res in results:
                try:
                    content_str = res.get("content", "{}")
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

            return {
                "point_adjustment": avg_adj,
                "uncertainty_factor": avg_uncertainty,
                "risk_level": final_risk,
                "reasoning": " | ".join(reasonings[:3]),  # Limit to 3 for brevity
                "model_count": valid_count,
            }

        except Exception as e:
            logger.error(f"Failed to aggregate consensus results: {e}")
            return {"error": str(e), "fallback": True}

    def _construct_prompt(self, context: dict[str, Any]) -> str:
        team1 = context.get("team1", "Team A")
        team2 = context.get("team2", "Team B")
        predicted_total = context.get("predicted_total", "N/A")
        market_line = context.get("market_line", "N/A")

        prompt_lines = [
            "You are a professional 'Sharp' NBA Bettor and Risk Analyst.",
            "Your goal is to identify Market Inefficiencies properly weighted by Uncertainty.",
            "",
            f"Matchup: {team1} vs {team2}",
            "",
            "## Quantitative Baseline",
            f"- Model Prediction: {predicted_total} pts",
            f"- Market Line: {market_line}",
            f"- Deviation: {context.get('deviation_from_market', 'N/A')}",
            f"- Model Confidence: {context.get('confidence', 'N/A')}",
            f"- Stats Key: {json.dumps(context.get('stats', {}), indent=2)}",
            "",
            "## News & Context",
        ]

        news = context.get("news", [])
        if news:
            for item in news[:12]:
                prompt_lines.append(
                    f"- [{item.get('type', 'news').upper()}]: {item.get('text', '')}"
                )
        else:
            prompt_lines.append("- No significant recent news found.")

        prompt_lines.append("")
        prompt_lines.append("## Task: Hybrid Evaluation (Bias + Variance)")
        prompt_lines.append(
            "1. **Point Adjustment (Bias)**: Based on NEWS/CONTEXT, should the Total be Higher or Lower than the Model/Market line?"
        )
        prompt_lines.append(
            "   - E.g. 'Star out -> -3.0 pts', 'Pace up -> +2.0 pts'. If priced in, use 0."
        )
        prompt_lines.append(
            "2. **Uncertainty Factor (Variance)**: How volatile/unpredictable is this game?"
        )
        prompt_lines.append("   - 0.0 = Extremely Stable (Full Confidence)")
        prompt_lines.append("   - 1.0 = Pure Chaos/Gambling (Zero Confidence)")
        prompt_lines.append("")
        prompt_lines.append("## Output Format (JSON ONLY)")
        prompt_lines.append("Return valid JSON with these exact keys:")
        prompt_lines.append(
            "- 'point_adjustment': (float) The proposed adjustment to the Total (Max +/- 15)."
        )
        prompt_lines.append(
            "- 'uncertainty_factor': (float) 0.0 to 1.0. YOUR ESTIMATE OF VOLATILITY."
        )
        prompt_lines.append("- 'risk_level': 'LOW', 'MED', 'HIGH'.")
        prompt_lines.append(
            "- 'reasoning': (string) Concise analysis justifying BOTH the adjustment and the uncertainty."
        )

        return "\n".join(prompt_lines)
