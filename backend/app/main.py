from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.logic import compute_next_generation

app = FastAPI(title="Conway's Game of Life Benchmark Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _empty_grid(rows: int, cols: int) -> np.ndarray:
    safe_rows = max(1, min(int(rows), 512))
    safe_cols = max(1, min(int(cols), 512))
    return np.zeros((safe_rows, safe_cols), dtype=np.uint8)


def _coerce_grid(raw_grid: Any, rows: int, cols: int) -> np.ndarray:
    try:
        data = np.asarray(raw_grid, dtype=np.uint8)
        if data.shape != (rows, cols):
            return _empty_grid(rows, cols)
        return (data > 0).astype(np.uint8)
    except Exception:
        return _empty_grid(rows, cols)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/simulate")
async def simulate(websocket: WebSocket) -> None:
    await websocket.accept()
    grid = _empty_grid(64, 64)

    try:
        while True:
            raw_payload = await websocket.receive_text()
            payload = json.loads(raw_payload)
            action = str(payload.get("action", "step")).lower()

            if action == "init":
                rows = int(payload.get("rows", grid.shape[0]))
                cols = int(payload.get("cols", grid.shape[1]))
                grid = _coerce_grid(payload.get("grid"), rows, cols)

                await websocket.send_json(
                    {
                        "type": "init",
                        "rows": int(grid.shape[0]),
                        "cols": int(grid.shape[1]),
                        "grid": grid.tolist(),
                    }
                )
                continue

            mode = str(payload.get("mode", "cpu")).lower()

            if action == "benchmark":
                steps = max(1, min(int(payload.get("steps", 100)), 10000))
                start = time.perf_counter()
                mode_used = mode
                for _ in range(steps):
                    grid, mode_used = compute_next_generation(grid, mode)
                total_ms = (time.perf_counter() - start) * 1000

                await websocket.send_json(
                    {
                        "type": "benchmark",
                        "modeRequested": mode,
                        "modeUsed": mode_used,
                        "steps": steps,
                        "totalMs": total_ms,
                        "avgMs": total_ms / steps,
                        "grid": grid.tolist(),
                    }
                )
                continue

            step_start = time.perf_counter()
            grid, mode_used = compute_next_generation(grid, mode)
            duration_ms = (time.perf_counter() - step_start) * 1000

            await websocket.send_json(
                {
                    "type": "step",
                    "modeRequested": mode,
                    "modeUsed": mode_used,
                    "durationMs": duration_ms,
                    "grid": grid.tolist(),
                }
            )

    except (WebSocketDisconnect, RuntimeError, ValueError, json.JSONDecodeError):
        return
