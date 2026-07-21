import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const DEFAULT_ROWS = 50
const DEFAULT_COLS = 80
const CELL_SIZE = 10

const createEmptyGrid = (rows, cols) =>
  Array.from({ length: rows }, () => Array(cols).fill(0))

const getWebSocketUrl = () => {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL
  }

  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const hostname = window.location.hostname || 'localhost'
  return `${protocol}://${hostname}:8000/ws/simulate`
}

function App() {
  const [rows, setRows] = useState(DEFAULT_ROWS)
  const [cols, setCols] = useState(DEFAULT_COLS)
  const [grid, setGrid] = useState(() => createEmptyGrid(DEFAULT_ROWS, DEFAULT_COLS))
  const [logicMode, setLogicMode] = useState('cpu')
  const [benchmarkSteps, setBenchmarkSteps] = useState(200)
  const [running, setRunning] = useState(false)
  const [connected, setConnected] = useState(false)
  const [lastStepMs, setLastStepMs] = useState(null)
  const [lastModeUsed, setLastModeUsed] = useState('cpu')
  const [benchmarkResult, setBenchmarkResult] = useState(null)

  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const drawingRef = useRef(false)
  const drawValueRef = useRef(1)

  const canvasWidth = useMemo(() => cols * CELL_SIZE, [cols])
  const canvasHeight = useMemo(() => rows * CELL_SIZE, [rows])

  useEffect(() => {
    const ws = new WebSocket(getWebSocketUrl())
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      ws.send(
        JSON.stringify({
          action: 'init',
          rows,
          cols,
          grid,
        }),
      )
    }

    ws.onclose = () => {
      setConnected(false)
      setRunning(false)
    }

    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data)
      if (payload.grid) {
        setGrid(payload.grid)
      }

      if (payload.type === 'step') {
        setLastStepMs(payload.durationMs)
        setLastModeUsed(payload.modeUsed)
      }

      if (payload.type === 'benchmark') {
        setBenchmarkResult(payload)
        setLastModeUsed(payload.modeUsed)
      }
    }

    return () => {
      ws.close()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!running || !connected || !wsRef.current) {
      return undefined
    }

    const interval = window.setInterval(() => {
      wsRef.current?.send(
        JSON.stringify({
          action: 'step',
          mode: logicMode,
        }),
      )
    }, 100)

    return () => window.clearInterval(interval)
  }, [running, connected, logicMode])

  useEffect(() => {
    const context = canvasRef.current?.getContext('2d')
    if (!context) {
      return
    }

    context.clearRect(0, 0, canvasWidth, canvasHeight)

    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        context.fillStyle = grid[row][col] ? '#16a34a' : '#111827'
        context.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
      }
    }

    context.strokeStyle = '#374151'
    context.lineWidth = 0.5

    for (let x = 0; x <= canvasWidth; x += CELL_SIZE) {
      context.beginPath()
      context.moveTo(x, 0)
      context.lineTo(x, canvasHeight)
      context.stroke()
    }

    for (let y = 0; y <= canvasHeight; y += CELL_SIZE) {
      context.beginPath()
      context.moveTo(0, y)
      context.lineTo(canvasWidth, y)
      context.stroke()
    }
  }, [grid, rows, cols, canvasWidth, canvasHeight])

  const applyCellAtPointer = (event, forceValue = null) => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    const col = Math.floor(x / CELL_SIZE)
    const row = Math.floor(y / CELL_SIZE)

    if (row < 0 || row >= rows || col < 0 || col >= cols) {
      return
    }

    setGrid((previousGrid) => {
      const nextGrid = previousGrid.map((line) => [...line])
      const nextValue = forceValue == null ? (nextGrid[row][col] ? 0 : 1) : forceValue
      nextGrid[row][col] = nextValue
      return nextGrid
    })
  }

  const handlePointerDown = (event) => {
    drawingRef.current = true

    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const rect = canvas.getBoundingClientRect()
    const col = Math.floor((event.clientX - rect.left) / CELL_SIZE)
    const row = Math.floor((event.clientY - rect.top) / CELL_SIZE)

    if (row < 0 || row >= rows || col < 0 || col >= cols) {
      return
    }

    const currentValue = grid[row][col]
    drawValueRef.current = currentValue ? 0 : 1
    applyCellAtPointer(event, drawValueRef.current)
  }

  const handlePointerMove = (event) => {
    if (!drawingRef.current) {
      return
    }

    applyCellAtPointer(event, drawValueRef.current)
  }

  const handlePointerUp = () => {
    drawingRef.current = false
  }

  const handleApplyGrid = () => {
    const nextRows = Math.max(1, Math.min(512, Number(rows) || DEFAULT_ROWS))
    const nextCols = Math.max(1, Math.min(512, Number(cols) || DEFAULT_COLS))
    const nextGrid = createEmptyGrid(nextRows, nextCols)

    setRows(nextRows)
    setCols(nextCols)
    setGrid(nextGrid)
    setRunning(false)

    wsRef.current?.send(
      JSON.stringify({
        action: 'init',
        rows: nextRows,
        cols: nextCols,
        grid: nextGrid,
      }),
    )
  }

  const handleClear = () => {
    const nextGrid = createEmptyGrid(rows, cols)
    setGrid(nextGrid)
    setRunning(false)

    wsRef.current?.send(
      JSON.stringify({
        action: 'init',
        rows,
        cols,
        grid: nextGrid,
      }),
    )
  }

  const handleStep = () => {
    wsRef.current?.send(
      JSON.stringify({
        action: 'init',
        rows,
        cols,
        grid,
      }),
    )
    wsRef.current?.send(
      JSON.stringify({
        action: 'step',
        mode: logicMode,
      }),
    )
  }

  const handleBenchmark = () => {
    wsRef.current?.send(
      JSON.stringify({
        action: 'init',
        rows,
        cols,
        grid,
      }),
    )

    wsRef.current?.send(
      JSON.stringify({
        action: 'benchmark',
        mode: logicMode,
        steps: Number(benchmarkSteps) || 1,
      }),
    )
  }

  return (
    <main className="app">
      <h1>Conway&apos;s Game of Life Benchmark</h1>

      <section className="controls">
        <label>
          Rows
          <input type="number" min="1" max="512" value={rows} onChange={(event) => setRows(event.target.value)} />
        </label>
        <label>
          Cols
          <input type="number" min="1" max="512" value={cols} onChange={(event) => setCols(event.target.value)} />
        </label>
        <button type="button" onClick={handleApplyGrid}>
          Apply Grid
        </button>
        <button type="button" onClick={handleClear}>
          Clear
        </button>

        <label>
          Logic
          <select value={logicMode} onChange={(event) => setLogicMode(event.target.value)}>
            <option value="cpu">CPU (NumPy)</option>
            <option value="gpu">GPU (CUDA)</option>
          </select>
        </label>

        <button type="button" onClick={handleStep} disabled={!connected}>
          Step
        </button>

        <button type="button" onClick={() => setRunning((value) => !value)} disabled={!connected}>
          {running ? 'Pause' : 'Run'}
        </button>

        <label>
          Benchmark Steps
          <input
            type="number"
            min="1"
            max="10000"
            value={benchmarkSteps}
            onChange={(event) => setBenchmarkSteps(event.target.value)}
          />
        </label>

        <button type="button" onClick={handleBenchmark} disabled={!connected}>
          Benchmark
        </button>
      </section>

      <p className="status">
        WebSocket: {connected ? 'connected' : 'disconnected'} | Mode used: {lastModeUsed}
        {lastStepMs != null ? ` | Last step: ${lastStepMs.toFixed(3)} ms` : ''}
      </p>

      {benchmarkResult ? (
        <p className="status">
          Benchmark ({benchmarkResult.modeUsed}, {benchmarkResult.steps} steps): total{' '}
          {benchmarkResult.totalMs.toFixed(2)} ms, avg {benchmarkResult.avgMs.toFixed(4)} ms/step
        </p>
      ) : null}

      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        className="life-canvas"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      />
    </main>
  )
}

export default App
