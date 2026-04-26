#!/usr/bin/env python3
"""
Central WebSocket server for the Energy Intelligence Dashboard.
Receives commands from dashboard.html and dispatches to oil_agent_v2 or ho_agent.
"""

import asyncio
import json
import traceback

# ── AGENT IMPORTS ─────────────────────────────────────────────────────────────

def _import_agents():
    agents = {}
    try:
        import oil_agent_v2 as oil
        agents["oil"] = oil
    except Exception as e:
        print("[WARN] Could not import oil_agent_v2: {}".format(e))
    try:
        import ho_agent as ho
        agents["ho"] = ho
    except Exception as e:
        print("[WARN] Could not import ho_agent: {}".format(e))
    return agents

AGENTS = _import_agents()

# ── WEBSOCKET HANDLER ─────────────────────────────────────────────────────────

async def handler(websocket):
    print("[WS] Client connected: {}".format(websocket.remote_address))
    try:
        async for raw_msg in websocket:
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            cmd = msg.get("command", "")
            print("[WS] Command: {}".format(cmd))

            if cmd == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

            elif cmd == "run_oil":
                await run_agent(websocket, "oil")

            elif cmd == "run_ho":
                await run_agent(websocket, "ho")

            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Unknown command: {}".format(cmd)
                }))

    except Exception as e:
        print("[WS] Handler error: {}".format(e))


async def run_agent(websocket, agent_key):
    """Run an agent and stream status updates, then send the final result."""
    agent_mod = AGENTS.get(agent_key)
    if not agent_mod:
        await websocket.send(json.dumps({
            "type": "error",
            "message": "Agent '{}' not available. Check imports.".format(agent_key)
        }))
        return

    await websocket.send(json.dumps({"type": "status", "message": "Starting {} agent...".format(agent_key.upper())}))

    loop = asyncio.get_event_loop()

    def send_status(msg):
        """Thread-safe status sender called from synchronous agent code."""
        future = asyncio.run_coroutine_threadsafe(
            websocket.send(json.dumps({"type": "status", "message": str(msg)})),
            loop
        )
        try:
            future.result(timeout=5)
        except Exception:
            pass

    try:
        # Run blocking agent in a thread executor so it doesn't block the event loop
        result = await loop.run_in_executor(
            None,
            lambda: agent_mod.run(send=send_status)
        )

        await websocket.send(json.dumps({
            "type": "result",
            "agent": agent_key,
            "data": result
        }))

    except Exception as e:
        err = traceback.format_exc()
        print("[WS] Agent error:\n{}".format(err))
        await websocket.send(json.dumps({
            "type": "error",
            "message": "Agent error: {}\n{}".format(str(e), err)
        }))


# ── MAIN ──────────────────────────────────────────────────────────────────────

HOST = "localhost"
PORT = 8765

async def main():
    print("=" * 50)
    print("  Energy Intelligence Dashboard — Server")
    print("  WebSocket: ws://{}:{}".format(HOST, PORT))
    print("  Agents loaded: {}".format(list(AGENTS.keys())))
    print("  Open dashboard.html in your browser to start.")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
