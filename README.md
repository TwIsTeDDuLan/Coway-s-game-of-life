# Coway-s-game-of-life

Interactive Conway's Game of Life benchmark with:
- **React frontend** canvas for drawing live cells
- **FastAPI websocket backend** for simulation updates
- **CPU** next-generation logic using **NumPy**
- **GPU** next-generation logic using **CUDA (Numba)** with automatic CPU fallback when CUDA is unavailable

## Run backend

```bash
cd /home/runner/work/Coway-s-game-of-life/Coway-s-game-of-life/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run frontend

```bash
cd /home/runner/work/Coway-s-game-of-life/Coway-s-game-of-life/frontend
npm install
npm run dev -- --host
```

By default, the frontend connects to `ws://<host>:8000/ws/simulate`.
You can override this with `VITE_WS_URL`.

## Frontend features

- Set **grid rows and columns**
- Draw/erase live cells directly on the canvas
- Choose simulation logic: **CPU (NumPy)** or **GPU (CUDA)**
- Run/pause continuous simulation updates via websocket
- Run benchmark for a fixed number of generations and view total/average time
