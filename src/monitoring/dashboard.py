from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from src.utils.config import DASHBOARD_HOST, DASHBOARD_PORT
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.trading.paper_trader import PaperTrader


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


def create_app(trader: "PaperTrader | None" = None) -> FastAPI:
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
        return _build_status(trader)

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
        asyncio.create_task(_broadcast_loop(trader))

    return app


async def _broadcast_loop(trader: "PaperTrader | None", interval: int = 5) -> None:
    while True:
        try:
            data = _build_status(trader)
            await _manager.broadcast(data)
        except Exception as exc:
            logger.debug(f"broadcast error: {exc}")
        await asyncio.sleep(interval)


def _build_status(trader: "PaperTrader | None") -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    if trader is None:
        return {"ts": ts, "equity": 0, "positions": [], "halted": False}

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
        "ts":             ts,
        "equity":         s["equity"],
        "peak_equity":    s["peak_equity"],
        "drawdown":       round(s["drawdown"] * 100, 2),
        "daily_pnl":      s["daily_pnl"],
        "open_positions": s["open_positions"],
        "total_exposure": s["total_exposure"],
        "halted":         s["halted"],
        "total_trades":   len(trader.closed_trades),
        "positions":      positions,
    }


async def run_dashboard(trader: "PaperTrader | None" = None) -> None:
    app = create_app(trader)
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
  header { background: #161b22; padding: 16px 24px; border-bottom: 1px solid #30363d;
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.2rem; font-weight: 600; }
  .badge { padding: 3px 8px; border-radius: 12px; font-size: .75rem; font-weight: 600; }
  .badge-paper { background: #1f6feb; color: #fff; }
  .badge-halted { background: #da3633; color: #fff; }
  .badge-ok { background: #238636; color: #fff; }
  #ts { margin-left: auto; font-size: .75rem; color: #8b949e; }
  main { padding: 24px; display: grid; gap: 20px; max-width: 1200px; margin: 0 auto; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card label { font-size: .7rem; color: #8b949e; text-transform: uppercase; letter-spacing: .05em; display: block; margin-bottom: 6px; }
  .card .val { font-size: 1.4rem; font-weight: 700; }
  .green { color: #3fb950; } .red { color: #f85149; } .yellow { color: #d29922; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; }
  th { text-align: left; padding: 8px 12px; background: #161b22;
       color: #8b949e; font-weight: 500; border-bottom: 1px solid #30363d; }
  td { padding: 7px 12px; border-bottom: 1px solid #21262d; }
  tr:hover td { background: #161b22; }
  .section-title { font-size: .9rem; font-weight: 600; color: #8b949e; margin-bottom: 10px; }
  canvas { max-height: 200px; }
</style>
</head>
<body>
<header>
  <h1>&#x1F4C8; Trading Dashboard</h1>
  <span class="badge badge-paper">PAPER</span>
  <span id="halt-badge" class="badge badge-ok" style="display:none">HALTED</span>
  <span id="ts"></span>
</header>
<main>
  <div class="cards">
    <div class="card"><label>Equity (USDT)</label><div class="val" id="equity">—</div></div>
    <div class="card"><label>Drawdown</label><div class="val" id="dd">—</div></div>
    <div class="card"><label>Daily PnL</label><div class="val" id="daily-pnl">—</div></div>
    <div class="card"><label>Open Positions</label><div class="val" id="n-pos">—</div></div>
    <div class="card"><label>Total Trades</label><div class="val" id="n-trades">—</div></div>
    <div class="card"><label>Exposure (USDT)</label><div class="val" id="exposure">—</div></div>
  </div>

  <div>
    <div class="section-title">Open Positions</div>
    <table id="pos-table">
      <thead><tr>
        <th>Symbol</th><th>Side</th><th>Entry</th><th>Size</th>
        <th>Stop Loss</th><th>Take Profit</th><th>Held</th>
      </tr></thead>
      <tbody id="pos-body"><tr><td colspan="7" style="color:#8b949e;padding:16px">No open positions</td></tr></tbody>
    </table>
  </div>

  <div>
    <div class="section-title">Recent Trades</div>
    <table id="trades-table">
      <thead><tr>
        <th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th>
        <th>PnL USDT</th><th>PnL %</th><th>Reason</th><th>Time</th>
      </tr></thead>
      <tbody id="trades-body"><tr><td colspan="8" style="color:#8b949e;padding:16px">Loading...</td></tr></tbody>
    </table>
  </div>
</main>

<script>
const fmt = (n, d=2) => n == null ? '—' : n.toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d});
const pct = n => n == null ? '—' : (n >= 0 ? '+' : '') + fmt(n) + '%';
const cls = n => n >= 0 ? 'green' : 'red';

function applyStatus(d) {
  document.getElementById('ts').textContent = new Date(d.ts).toLocaleTimeString();
  document.getElementById('equity').textContent = fmt(d.equity);
  const ddEl = document.getElementById('dd');
  ddEl.textContent = pct(d.drawdown);
  ddEl.className = 'val ' + (d.drawdown > 5 ? 'red' : d.drawdown > 2 ? 'yellow' : 'green');
  const dpEl = document.getElementById('daily-pnl');
  dpEl.textContent = (d.daily_pnl >= 0 ? '+' : '') + fmt(d.daily_pnl);
  dpEl.className = 'val ' + cls(d.daily_pnl);
  document.getElementById('n-pos').textContent = d.open_positions;
  document.getElementById('n-trades').textContent = d.total_trades;
  document.getElementById('exposure').textContent = fmt(d.total_exposure);
  const hb = document.getElementById('halt-badge');
  hb.style.display = d.halted ? 'inline' : 'none';

  // Positions table
  const pb = document.getElementById('pos-body');
  if (!d.positions || d.positions.length === 0) {
    pb.innerHTML = '<tr><td colspan="7" style="color:#8b949e;padding:16px">No open positions</td></tr>';
  } else {
    pb.innerHTML = d.positions.map(p => `<tr>
      <td><b>${p.symbol}</b></td>
      <td style="color:${p.direction==='LONG'?'#3fb950':'#f85149'}">${p.direction}</td>
      <td>${fmt(p.entry_price, 4)}</td>
      <td>${fmt(p.notional, 2)}</td>
      <td>${fmt(p.stop_loss, 4)}</td>
      <td>${fmt(p.take_profit, 4)}</td>
      <td>${p.held_hours}h</td>
    </tr>`).join('');
  }
}

async function loadTrades() {
  try {
    const r = await fetch('/api/trades');
    const d = await r.json();
    const tb = document.getElementById('trades-body');
    if (!d.trades || d.trades.length === 0) {
      tb.innerHTML = '<tr><td colspan="8" style="color:#8b949e;padding:16px">No trades yet</td></tr>';
      return;
    }
    tb.innerHTML = d.trades.map(t => {
      const sign = t.pnl_usd >= 0 ? '+' : '';
      const c = t.pnl_usd >= 0 ? 'green' : 'red';
      const dt = new Date(t.exit_time).toLocaleString('en-US',{month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'});
      return `<tr>
        <td><b>${t.symbol}</b></td>
        <td style="color:${t.direction==='LONG'?'#3fb950':'#f85149'}">${t.direction}</td>
        <td>${fmt(t.entry_price, 4)}</td>
        <td>${fmt(t.exit_price, 4)}</td>
        <td class="${c}">${sign}${fmt(t.pnl_usd)}</td>
        <td class="${c}">${sign}${fmt(t.pnl_pct)}%</td>
        <td><span style="font-size:.75rem;color:#8b949e">${t.reason}</span></td>
        <td style="font-size:.75rem;color:#8b949e">${dt}</td>
      </tr>`;
    }).join('');
  } catch(e) { console.error(e); }
}

// WebSocket live updates
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = e => {
    try { applyStatus(JSON.parse(e.data)); } catch(_) {}
  };
  ws.onclose = () => setTimeout(connect, 3000);
  // Ping every 20s to keep alive
  setInterval(() => { if (ws.readyState === 1) ws.send('ping'); }, 20000);
}

// Initial load
fetch('/api/status').then(r => r.json()).then(applyStatus).catch(console.error);
loadTrades();
setInterval(loadTrades, 15000);
connect();
</script>
</body>
</html>
"""
