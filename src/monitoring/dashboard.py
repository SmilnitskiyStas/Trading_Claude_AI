from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import io
import time

import pandas as pd
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import text
import uvicorn

from src.utils.config import ANTHROPIC_API_KEY, DASHBOARD_HOST, DASHBOARD_PORT
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.trading.paper_trader import PaperTrader
    from src.ai_agent.agent import TradingAgent


class ConnectionManager:
    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws) if hasattr(self._clients, "discard") else None
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, data: dict) -> None:
        dead = []
        for client in self._clients:
            try:
                await client.send_text(json.dumps(data))
            except Exception:
                dead.append(client)
        for d in dead:
            self.disconnect(d)


_manager = ConnectionManager()


def create_app(
    trader: "PaperTrader | None" = None,
    agent: "TradingAgent | None" = None,
) -> FastAPI:
    app = FastAPI(title="Trading Dashboard", docs_url=None, redoc_url=None)

    # ── REST endpoints ─────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_HTML)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/status")
    async def api_status() -> dict:
        return _build_status(trader, agent)

    @app.get("/api/trades")
    async def api_trades() -> dict:
        if trader is None:
            return {"trades": []}
        trades = [
            {
                "symbol":      t.symbol,
                "direction":   "LONG" if t.direction > 0 else "SHORT",
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "pnl_usd":     round(t.pnl_usd, 2),
                "pnl_pct":     round(t.pnl_pct * 100, 2),
                "entry_time":  t.entry_time.isoformat(),
                "exit_time":   t.exit_time.isoformat(),
                "reason":      t.exit_reason,
            }
            for t in reversed(trader.closed_trades[-50:])
        ]
        return {"trades": trades}

    @app.get("/api/equity")
    async def api_equity() -> dict:
        if trader is None:
            return {"curve": []}
        return {"curve": trader.equity_curve[-200:]}

    # ── OHLCV upload ──────────────────────────────────────────────────────

    @app.post("/api/upload/ohlcv")
    async def upload_ohlcv(file: UploadFile = File(...)) -> JSONResponse:
        """Accept CSV file and bulk-insert into ohlcv_data table."""
        from src.utils.database import AsyncSessionFactory
        t0 = time.time()
        try:
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content))

            required = {"exchange", "symbol", "timeframe", "timestamp"}
            missing = required - set(df.columns)
            if missing:
                return JSONResponse(status_code=400,
                                    content={"error": f"Missing columns: {missing}"})

            if "id" in df.columns:
                df = df.drop(columns=["id"])
            df = df.where(pd.notnull(df), None)

            records = df.to_dict("records")
            insert_sql = text("""
                INSERT INTO ohlcv_data
                    (exchange, symbol, timeframe, timestamp,
                     open, high, low, close, volume)
                VALUES
                    (:exchange, :symbol, :timeframe, :timestamp,
                     :open, :high, :low, :close, :volume)
                ON CONFLICT (exchange, symbol, timeframe, timestamp) DO NOTHING
            """)

            async with AsyncSessionFactory() as session:
                await session.execute(text("TRUNCATE ohlcv_data"))
                chunk = 5000
                for i in range(0, len(records), chunk):
                    await session.execute(insert_sql, records[i : i + chunk])
                await session.commit()

            duration = round(time.time() - t0, 1)
            logger.info(f"OHLCV upload: {len(df)} rows in {duration}s ({file.filename})")
            return JSONResponse(content={
                "status": "ok", "rows": len(df), "duration_sec": duration
            })
        except Exception as exc:
            logger.error(f"OHLCV upload failed: {exc}")
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.get("/api/db/stats")
    async def api_db_stats() -> dict:
        """Return row counts per exchange/symbol/timeframe."""
        from src.utils.database import AsyncSessionFactory
        try:
            async with AsyncSessionFactory() as session:
                result = await session.execute(text("""
                    SELECT exchange, symbol, timeframe, COUNT(*) AS cnt
                    FROM ohlcv_data
                    GROUP BY 1, 2, 3
                    ORDER BY 1, 2, 3
                """))
                rows = [{"exchange": r[0], "symbol": r[1],
                         "timeframe": r[2], "count": int(r[3])}
                        for r in result.fetchall()]
            return {"rows": rows, "total": sum(r["count"] for r in rows)}
        except Exception as exc:
            return {"rows": [], "total": 0, "error": str(exc)}

    # ── AI Agent toggle ────────────────────────────────────────────────────

    @app.get("/api/agent")
    async def api_agent_status() -> dict:
        api_key_set = bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY not in ("your_key", ""))
        return {
            "available":   agent is not None,
            "enabled":     agent.enabled if agent else False,
            "api_key_set": api_key_set,
        }

    @app.post("/api/agent/enable")
    async def api_agent_enable() -> dict:
        if agent is None:
            return {"ok": False, "error": "Agent not initialized (check ANTHROPIC_API_KEY in .env)"}
        agent.enabled = True
        logger.info("AI agent ENABLED via dashboard")
        return {"ok": True, "enabled": True}

    @app.post("/api/agent/disable")
    async def api_agent_disable() -> dict:
        if agent is None:
            return {"ok": False, "error": "Agent not available"}
        agent.enabled = False
        logger.info("AI agent DISABLED via dashboard")
        return {"ok": True, "enabled": False}

    # ── WebSocket ──────────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await _manager.connect(ws)
        try:
            while True:
                await ws.receive_text()   # keep-alive; client sends pings
        except WebSocketDisconnect:
            _manager.disconnect(ws)

    # ── Background broadcaster ─────────────────────────────────────────────

    @app.on_event("startup")
    async def _start_broadcaster() -> None:
        asyncio.create_task(_broadcast_loop(trader, agent))

    return app


async def _broadcast_loop(
    trader: "PaperTrader | None",
    agent: "TradingAgent | None" = None,
    interval: int = 5,
) -> None:
    while True:
        try:
            data = _build_status(trader, agent)
            await _manager.broadcast(data)
        except Exception as exc:
            logger.debug(f"broadcast error: {exc}")
        await asyncio.sleep(interval)


def _build_status(
    trader: "PaperTrader | None",
    agent: "TradingAgent | None" = None,
) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    if trader is None:
        return {
            "ts": ts, "equity": 0, "positions": [], "halted": False,
            "agent_available": agent is not None,
            "agent_enabled":   agent.enabled if agent else False,
        }

    rm = trader.rm
    s = rm.summary()

    positions = [
        {
            "symbol":      sym,
            "direction":   "LONG" if pos.direction > 0 else "SHORT",
            "entry_price": pos.entry_price,
            "notional":    pos.notional_usd,
            "stop_loss":   pos.stop_loss,
            "take_profit": pos.take_profit,
            "held_hours":  round(
                (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600, 1
            ),
        }
        for sym, pos in trader._positions.items()
    ]

    return {
        "ts":              ts,
        "equity":          s["equity"],
        "peak_equity":     s["peak_equity"],
        "drawdown":        round(s["drawdown"] * 100, 2),
        "daily_pnl":       s["daily_pnl"],
        "open_positions":  s["open_positions"],
        "total_exposure":  s["total_exposure"],
        "halted":          s["halted"],
        "total_trades":    len(trader.closed_trades),
        "positions":       positions,
        "agent_available": agent is not None,
        "agent_enabled":   agent.enabled if agent else False,
    }


async def run_dashboard(
    trader: "PaperTrader | None" = None,
    agent: "TradingAgent | None" = None,
) -> None:
    app = create_app(trader, agent)
    config = uvicorn.Config(
        app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    logger.info(f"Dashboard starting on http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    await server.serve()


# ── Embedded HTML dashboard ────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', monospace; background: #0d1117; color: #c9d1d9; min-height: 100vh; }
  header { background: #161b22; padding: 14px 24px; border-bottom: 1px solid #30363d;
           display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  header h1 { font-size: 1.15rem; font-weight: 600; }
  .badge { padding: 3px 8px; border-radius: 12px; font-size: .72rem; font-weight: 600; }
  .badge-paper  { background: #1f6feb; color: #fff; }
  .badge-halted { background: #da3633; color: #fff; }
  #ts { margin-left: auto; font-size: .72rem; color: #8b949e; }
  main { padding: 20px 24px; display: grid; gap: 20px; max-width: 1200px; margin: 0 auto; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 16px; }
  .card label { font-size: .68rem; color: #8b949e; text-transform: uppercase;
                letter-spacing: .05em; display: block; margin-bottom: 5px; }
  .card .val { font-size: 1.35rem; font-weight: 700; }
  .green  { color: #3fb950; }
  .red    { color: #f85149; }
  .yellow { color: #d29922; }
  .gray   { color: #8b949e; }

  /* ── Agent panel ──────────────────────────────────────────── */
  .agent-panel {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 16px 20px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }
  .agent-panel .agent-info { flex: 1; min-width: 200px; }
  .agent-panel .agent-title {
    font-size: .85rem; font-weight: 600; margin-bottom: 4px; display: flex;
    align-items: center; gap: 8px;
  }
  .agent-panel .agent-desc { font-size: .75rem; color: #8b949e; line-height: 1.4; }
  .agent-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
  .dot-on   { background: #3fb950; box-shadow: 0 0 6px #3fb950; }
  .dot-off  { background: #8b949e; }
  .dot-na   { background: #da3633; }

  /* Toggle switch */
  .toggle-wrap { display: flex; align-items: center; gap: 10px; }
  .toggle-label { font-size: .78rem; color: #8b949e; min-width: 55px; }
  .switch { position: relative; display: inline-block; width: 46px; height: 24px; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider {
    position: absolute; cursor: pointer; inset: 0;
    background: #30363d; border-radius: 24px; transition: .25s;
  }
  .slider:before {
    position: absolute; content: ""; height: 18px; width: 18px;
    left: 3px; bottom: 3px; background: #fff; border-radius: 50%; transition: .25s;
  }
  input:checked + .slider { background: #1f6feb; }
  input:checked + .slider:before { transform: translateX(22px); }
  input:disabled + .slider { opacity: .4; cursor: not-allowed; }
  .toggle-status { font-size: .78rem; font-weight: 600; min-width: 60px; }

  /* ── Upload panel ────────────────────────────────────────── */
  .upload-panel {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 18px 20px;
  }
  .upload-desc { font-size: .78rem; color: #8b949e; margin: 6px 0 14px; line-height: 1.5; }
  .upload-steps { background: #0d1117; border: 1px solid #21262d; border-radius: 6px;
                  padding: 12px 14px; margin-bottom: 14px; }
  .upload-steps .step-title { font-size: .72rem; color: #8b949e; margin-bottom: 6px; }
  .upload-steps code { font-size: .73rem; color: #79c0ff; white-space: pre; display: block;
                       line-height: 1.6; }
  .drop-zone {
    border: 2px dashed #30363d; border-radius: 8px; padding: 28px;
    text-align: center; cursor: pointer; transition: .2s;
    font-size: .85rem; color: #8b949e;
  }
  .drop-zone:hover, .drop-zone.over { border-color: #1f6feb; color: #79c0ff; background: #0d1117; }
  .progress-wrap { background: #21262d; border-radius: 4px; height: 6px; margin-top: 10px; overflow: hidden; }
  .progress-bar  { height: 100%; background: #1f6feb; width: 0%; transition: width .3s; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 6px; margin-top: 12px; }
  .stats-item { background: #0d1117; border: 1px solid #21262d; border-radius: 6px;
                padding: 8px 12px; font-size: .75rem; }
  .stats-item .si-sym  { font-weight: 600; color: #c9d1d9; }
  .stats-item .si-info { color: #8b949e; margin-top: 2px; }

  /* ── Tables ───────────────────────────────────────────────── */
  table { width: 100%; border-collapse: collapse; font-size: .83rem; }
  th { text-align: left; padding: 8px 12px; background: #161b22;
       color: #8b949e; font-weight: 500; border-bottom: 1px solid #30363d; }
  td { padding: 7px 12px; border-bottom: 1px solid #21262d; }
  tr:hover td { background: #161b22; }
  .section-title { font-size: .88rem; font-weight: 600; color: #8b949e; margin-bottom: 10px; }
</style>
</head>
<body>
<header>
  <h1>&#x1F4C8; Trading Dashboard</h1>
  <span class="badge badge-paper">PAPER</span>
  <span id="halt-badge" class="badge" style="display:none;background:#da3633;color:#fff">&#x26D4; HALTED</span>
  <span id="ts"></span>
</header>
<main>

  <!-- Portfolio cards -->
  <div class="cards">
    <div class="card"><label>Equity (USDT)</label><div class="val" id="equity">—</div></div>
    <div class="card"><label>Drawdown</label><div class="val" id="dd">—</div></div>
    <div class="card"><label>Daily PnL</label><div class="val" id="daily-pnl">—</div></div>
    <div class="card"><label>Open Positions</label><div class="val" id="n-pos">—</div></div>
    <div class="card"><label>Total Trades</label><div class="val" id="n-trades">—</div></div>
    <div class="card"><label>Exposure (USDT)</label><div class="val" id="exposure">—</div></div>
  </div>

  <!-- AI Agent control panel -->
  <div class="agent-panel">
    <div class="agent-info">
      <div class="agent-title">
        <span class="agent-dot dot-na" id="agent-dot"></span>
        &#x1F916; AI Agent (Claude)
      </div>
      <div class="agent-desc" id="agent-desc">
        Analyzes ML signals + news sentiment once per hour. Results cached in Redis.
        When enabled, Claude confirms or overrides ML signals before trade opens.
      </div>
    </div>
    <div class="toggle-wrap">
      <span class="toggle-label" id="agent-label">N/A</span>
      <label class="switch">
        <input type="checkbox" id="agent-toggle" disabled onchange="toggleAgent(this.checked)">
        <span class="slider"></span>
      </label>
      <span class="toggle-status gray" id="agent-status-text">—</span>
    </div>
  </div>

  <!-- OHLCV Data Upload -->
  <div class="upload-panel">
    <div class="section-title">&#x1F4C2; OHLCV Data Upload</div>
    <div class="upload-desc">
      Upload a CSV file exported from your local PostgreSQL to import OHLCV data into the server database.
      Existing data will be replaced.
    </div>
    <div class="upload-steps">
      <div class="step-title">Local export commands (Windows cmd.exe):</div>
      <code>docker compose exec postgres psql -U trader trading -c "\\COPY ohlcv_data TO '/tmp/ohlcv.csv' WITH (FORMAT csv, HEADER true)"
docker cp trading_postgres:/tmp/ohlcv.csv ohlcv.csv</code>
    </div>
    <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()"
         ondragover="event.preventDefault();this.classList.add('over')"
         ondragleave="this.classList.remove('over')"
         ondrop="this.classList.remove('over');handleFile(event.dataTransfer.files[0]);event.preventDefault()">
      <span id="drop-text">&#x1F5C2; Click or drag CSV file here</span>
    </div>
    <input type="file" id="file-input" accept=".csv" style="display:none"
           onchange="handleFile(this.files[0])">
    <div class="progress-wrap" id="progress-wrap" style="display:none">
      <div class="progress-bar" id="progress-bar"></div>
    </div>
    <div id="upload-status" style="font-size:.8rem;margin-top:8px;min-height:20px"></div>
    <div id="db-stats-wrap"></div>
  </div>

  <!-- Open positions -->
  <div>
    <div class="section-title">Open Positions</div>
    <table>
      <thead><tr>
        <th>Symbol</th><th>Side</th><th>Entry</th><th>Size USDT</th>
        <th>Stop Loss</th><th>Take Profit</th><th>Held</th>
      </tr></thead>
      <tbody id="pos-body">
        <tr><td colspan="7" style="color:#8b949e;padding:16px">No open positions</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Recent trades -->
  <div>
    <div class="section-title">Recent Trades</div>
    <table>
      <thead><tr>
        <th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th>
        <th>PnL USDT</th><th>PnL %</th><th>Reason</th><th>Closed</th>
      </tr></thead>
      <tbody id="trades-body">
        <tr><td colspan="8" style="color:#8b949e;padding:16px">Loading...</td></tr>
      </tbody>
    </table>
  </div>

</main>
<script>
const fmt = (n, d=2) => n == null ? '—' : n.toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d});
const cls  = n => n >= 0 ? 'green' : 'red';

// ── Portfolio status ───────────────────────────────────────────────────────
function applyStatus(d) {
  document.getElementById('ts').textContent = new Date(d.ts).toLocaleTimeString();
  document.getElementById('equity').textContent = fmt(d.equity);

  const ddEl = document.getElementById('dd');
  const ddVal = d.drawdown ?? 0;
  ddEl.textContent = (ddVal >= 0 ? '+' : '') + fmt(ddVal) + '%';
  ddEl.className = 'val ' + (ddVal > 5 ? 'red' : ddVal > 2 ? 'yellow' : 'green');

  const dpEl = document.getElementById('daily-pnl');
  const dp = d.daily_pnl ?? 0;
  dpEl.textContent = (dp >= 0 ? '+' : '') + fmt(dp);
  dpEl.className = 'val ' + cls(dp);

  document.getElementById('n-pos').textContent    = d.open_positions ?? '—';
  document.getElementById('n-trades').textContent = d.total_trades   ?? '—';
  document.getElementById('exposure').textContent = fmt(d.total_exposure);
  document.getElementById('halt-badge').style.display = d.halted ? 'inline' : 'none';

  // Positions table
  const pb = document.getElementById('pos-body');
  if (!d.positions || d.positions.length === 0) {
    pb.innerHTML = '<tr><td colspan="7" style="color:#8b949e;padding:14px">No open positions</td></tr>';
  } else {
    pb.innerHTML = d.positions.map(p => `<tr>
      <td><b>${p.symbol}</b></td>
      <td style="color:${p.direction==='LONG'?'#3fb950':'#f85149'}">${p.direction}</td>
      <td>${fmt(p.entry_price, 4)}</td>
      <td>${fmt(p.notional)}</td>
      <td>${fmt(p.stop_loss, 4)}</td>
      <td>${fmt(p.take_profit, 4)}</td>
      <td>${p.held_hours}h</td>
    </tr>`).join('');
  }

  // AI Agent panel
  applyAgentStatus(d.agent_available, d.agent_enabled);
}

// ── AI Agent panel ─────────────────────────────────────────────────────────
function applyAgentStatus(available, enabled) {
  const dot   = document.getElementById('agent-dot');
  const label = document.getElementById('agent-label');
  const tog   = document.getElementById('agent-toggle');
  const txt   = document.getElementById('agent-status-text');

  if (!available) {
    dot.className   = 'agent-dot dot-na';
    label.textContent = 'No API key';
    tog.disabled    = true;
    tog.checked     = false;
    txt.textContent = 'Unavailable';
    txt.className   = 'toggle-status gray';
    document.getElementById('agent-desc').textContent =
      'Set ANTHROPIC_API_KEY in .env and restart to enable the AI Agent.';
  } else if (enabled) {
    dot.className   = 'agent-dot dot-on';
    label.textContent = 'Enabled';
    tog.disabled    = false;
    tog.checked     = true;
    txt.textContent = 'Active';
    txt.className   = 'toggle-status green';
  } else {
    dot.className   = 'agent-dot dot-off';
    label.textContent = 'Disabled';
    tog.disabled    = false;
    tog.checked     = false;
    txt.textContent = 'Paused';
    txt.className   = 'toggle-status yellow';
  }
}

async function toggleAgent(enable) {
  const endpoint = enable ? '/api/agent/enable' : '/api/agent/disable';
  try {
    const r = await fetch(endpoint, { method: 'POST' });
    const d = await r.json();
    if (!d.ok) {
      alert('Agent toggle failed: ' + (d.error || 'unknown error'));
      // Revert toggle visually
      document.getElementById('agent-toggle').checked = !enable;
    } else {
      applyAgentStatus(true, d.enabled);
    }
  } catch(e) {
    alert('Request failed: ' + e);
    document.getElementById('agent-toggle').checked = !enable;
  }
}

// ── OHLCV Upload ───────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file) return;
  if (!file.name.endsWith('.csv')) {
    setUploadStatus('&#x274C; Only .csv files are supported', 'red'); return;
  }
  document.getElementById('drop-text').textContent = '&#x23F3; Uploading: ' + file.name;
  const prog = document.getElementById('progress-wrap');
  const bar  = document.getElementById('progress-bar');
  prog.style.display = 'block';
  bar.style.width = '0%';

  const form = new FormData();
  form.append('file', file);
  const xhr = new XMLHttpRequest();

  xhr.upload.onprogress = e => {
    if (e.lengthComputable) bar.style.width = Math.round(e.loaded / e.total * 80) + '%';
  };
  xhr.onload = () => {
    bar.style.width = '100%';
    try {
      const d = JSON.parse(xhr.responseText);
      if (d.error) {
        setUploadStatus('&#x274C; Error: ' + d.error, 'red');
      } else {
        setUploadStatus(
          `&#x2705; Imported <b>${d.rows.toLocaleString()}</b> rows in ${d.duration_sec}s`,
          'green'
        );
        loadDbStats();
      }
    } catch(e) { setUploadStatus('&#x274C; Server error', 'red'); }
    document.getElementById('drop-text').textContent = '&#x1F5C2; Click or drag CSV file here';
    document.getElementById('file-input').value = '';
  };
  xhr.onerror = () => {
    setUploadStatus('&#x274C; Upload failed', 'red');
    document.getElementById('drop-text').textContent = '&#x1F5C2; Click or drag CSV file here';
  };
  xhr.open('POST', '/api/upload/ohlcv');
  xhr.send(form);
  setUploadStatus('&#x23F3; Uploading ' + (file.size / 1024 / 1024).toFixed(1) + ' MB...', 'yellow');
}

function setUploadStatus(msg, color) {
  const el = document.getElementById('upload-status');
  el.innerHTML = msg;
  el.style.color = color === 'green' ? '#3fb950' : color === 'red' ? '#f85149' : '#d29922';
}

async function loadDbStats() {
  try {
    const r = await fetch('/api/db/stats');
    const d = await r.json();
    const wrap = document.getElementById('db-stats-wrap');
    if (!d.total) { wrap.innerHTML = ''; return; }
    const bySymbol = {};
    d.rows.forEach(row => {
      if (!bySymbol[row.symbol]) bySymbol[row.symbol] = [];
      bySymbol[row.symbol].push(row);
    });
    const items = Object.entries(bySymbol).map(([sym, rows]) => {
      const info = rows.map(r => `${r.timeframe}: ${r.count.toLocaleString()}`).join(' · ');
      return `<div class="stats-item">
        <div class="si-sym">${sym}</div>
        <div class="si-info">${info}</div>
      </div>`;
    }).join('');
    wrap.innerHTML = `
      <div style="font-size:.75rem;color:#8b949e;margin:12px 0 6px">
        Database: <b style="color:#c9d1d9">${d.total.toLocaleString()}</b> candles total
      </div>
      <div class="stats-grid">${items}</div>`;
  } catch(e) {}
}

// ── Recent trades ──────────────────────────────────────────────────────────
async function loadTrades() {
  try {
    const r = await fetch('/api/trades');
    const d = await r.json();
    const tb = document.getElementById('trades-body');
    if (!d.trades || d.trades.length === 0) {
      tb.innerHTML = '<tr><td colspan="8" style="color:#8b949e;padding:14px">No closed trades yet</td></tr>';
      return;
    }
    tb.innerHTML = d.trades.map(t => {
      const sign = t.pnl_usd >= 0 ? '+' : '';
      const c    = t.pnl_usd >= 0 ? 'green' : 'red';
      const dt   = new Date(t.exit_time).toLocaleString('en-US',
                     {month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit'});
      return `<tr>
        <td><b>${t.symbol}</b></td>
        <td style="color:${t.direction==='LONG'?'#3fb950':'#f85149'}">${t.direction}</td>
        <td>${fmt(t.entry_price, 4)}</td>
        <td>${fmt(t.exit_price, 4)}</td>
        <td class="${c}">${sign}${fmt(t.pnl_usd)}</td>
        <td class="${c}">${sign}${fmt(t.pnl_pct)}%</td>
        <td style="font-size:.72rem;color:#8b949e">${t.reason}</td>
        <td style="font-size:.72rem;color:#8b949e">${dt}</td>
      </tr>`;
    }).join('');
  } catch(e) { console.error(e); }
}

// ── WebSocket ──────────────────────────────────────────────────────────────
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = e => { try { applyStatus(JSON.parse(e.data)); } catch(_) {} };
  ws.onclose   = () => setTimeout(connect, 3000);
  setInterval(() => { if (ws.readyState === 1) ws.send('ping'); }, 20000);
}

// ── Init ───────────────────────────────────────────────────────────────────
fetch('/api/status').then(r => r.json()).then(applyStatus).catch(console.error);
loadTrades();
loadDbStats();
setInterval(loadTrades, 15000);
connect();
</script>
</body>
</html>
"""
