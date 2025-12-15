import sys
import json
import logging
import signal
import time
import os

# Setup logging to file with unbuffered output
logging.basicConfig(
    filename="mock_debug.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    force=True,
)


def log(msg):
    logging.info(msg)
    # Ensure logs are written immediately
    for handler in logging.getLogger().handlers:
        handler.flush()


def main():
    # Ignore SIGPIPE to prevent immediate termination if client closes pipe early
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except AttributeError:
        # Windows doesn't have SIGPIPE
        pass

    log("--- Mock Server Started ---")
    log(f"CWD: {os.getcwd()}")
    log(f"Args: {sys.argv}")

    # Set unbuffered stdout
    sys.stdout.reconfigure(line_buffering=True)

    buffer = ""
    while True:
        try:
            chunk = sys.stdin.read(1)
            if not chunk:
                # EOF on stdin usually means the parent process closed the pipe or died
                log("Stdin closed (EOF), exiting.")
                break

            buffer += chunk
            if buffer.endswith("\n"):
                line = buffer.strip()
                buffer = ""
                if not line:
                    continue

                log(f"Received: {line}")

                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    log(f"JSON Decode Error: {e}")
                    continue

                response = None

                if request.get("method") == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "capabilities": {},
                            "serverInfo": {"name": "mock-server", "version": "1.0"},
                        },
                    }
                elif request.get("method") == "tools/call":
                    # Check for delay mode
                    mode = os.environ.get("MOCK_MODE", "normal")
                    if mode == "timeout":
                        log("Simulating timeout (10s delay)...")
                        time.sleep(10)
                    else:
                        # consistent small delay to ensure client is ready to read
                        time.sleep(0.5)

                    # Mock Analysis Result
                    analysis = {
                        "consensus_score": 88,
                        "reasoning": "The mock expert indicates strong alignment with the quantitative model due to recent injury news favoring the under.",
                        "risk_level": "low",
                    }

                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(analysis)}],
                            "isError": False,
                        },
                    }

                if response:
                    output = json.dumps(response)
                    log(f"Sending: {output}")
                    try:
                        print(output)
                        sys.stdout.flush()
                        log("Sent and flushed.")
                    except BrokenPipeError:
                        log("BrokenPipeError caught during print/flush.")
                        break
                    except Exception as e:
                        log(f"Error sending response: {e}")

        except Exception as e:
            log(f"Unexpected error in main loop: {e}")
            break


if __name__ == "__main__":
    main()
