import asyncio
import json
import logging
import os
import sys
from typing import Optional, Any

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
        complexity: str = "simple",
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
        self, context: dict[str, Any], complexity: str = "simple"
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
            return asyncio.run(
                self._query_consensus_async(prompt, system_prompt, complexity)
            )
        except Exception as e:
            logger.error(f"❌ NanoGPT Consensus Sync Error: {e}")
            return {"error": str(e), "fallback": True}

    def _construct_prompt(self, context: dict[str, Any]) -> str:
        """
        Formats the context dictionary into a clear text prompt for the LLM.
        """
        team1 = context.get("team1", "Team A")
        team2 = context.get("team2", "Team B")

        prompt_lines = [
            f"Analyze the NBA matchup between {team1} and {team2}.",
            "## Quantitative Signal (The Quant Expert)",
            f"- Predicted Total: {context.get('predicted_total', 'N/A')}",
            f"- Model Confidence: {context.get('confidence', 'N/A')}",
            f"- Key Stats: {json.dumps(context.get('stats', {}), indent=2)}",
            "",
            "## Qualitative Signal (The News Feed)",
            "Recent News & Injuries:",
        ]

        news = context.get("news", [])
        if news:
            for item in news[:10]:  # Limit to top 10 relevant items
                prompt_lines.append(
                    f"- [{item.get('type', 'news').upper()}]: {item.get('text', '')}"
                )
        else:
            prompt_lines.append("- No significant recent news found.")

        prompt_lines.append("")
        prompt_lines.append("## Task")
        prompt_lines.append(
            "Provide a consensus analysis. Does the news support or contradict the quantitative model? Are there key player missing items that the stats model might miss?"
        )
        prompt_lines.append(
            "Return a valid JSON object with keys: 'consensus_score' (0-100), 'reasoning' (string), 'risk_level' (low/med/high)."
        )

        return "\n".join(prompt_lines)
