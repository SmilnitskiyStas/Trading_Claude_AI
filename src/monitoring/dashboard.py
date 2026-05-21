from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import io
import time

import pandas as pd
from fastapi import FastAPI, File, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import text
import uvicorn

from src.utils.config import (
    ANTHROPIC_API_KEY, DASHBOARD_HOST, DASHBOARD_PORT,
    DASHBOARD_USER, DASHBOARD_PASSWORD,
)
from src.utils.logger import logger

# ── Auth helpers ───────────────────────────────────────────────────────────
# Strategy:
#   1. Browser loads "/" with HTTP Basic Auth → dialog appears once
#   2. On success, a session cookie is set
#   3. All JS fetch() calls are authenticated via that cookie (no more dialogs)

_COOKIE_NAME = "_td_sess"


def _session_token() -> str:
    """Derive a fixed session token from the configured password (HMAC-SHA256)."""
    if not DASHBOARD_PASSWORD:
        return ""
    return hmac.new(
        DASHBOARD_PASSWORD.encode(), b"td_session_v1", hashlib.sha256
    ).hexdigest()


def _check_auth(request: Request) -> bool:
    """Return True when auth is disabled OR request carries valid cookie/Basic-Auth."""
    if not DASHBOARD_USER or not DASHBOARD_PASSWORD:
        return True                              # auth not configured → open access

    # ① Cookie (set after first successful Basic-Auth login — used by all JS calls)
    cookie = request.cookies.get(_COOKIE_NAME, "")
    if cookie and secrets.compare_digest(cookie, _session_token()):
        return True

    # ② HTTP Basic Auth (initial browser login)
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth[6:]).decode()
        user, _, password = decoded.partition(":")
    except Exception:
        return False
    return (
        secrets.compare_digest(user.encode(),     DASHBOARD_USER.encode()) and
        secrets.compare_digest(password.encode(), DASHBOARD_PASSWORD.encode())
    )


def _auth_required_response() -> Response:
    return Response(
        status_code=401,
        content="Unauthorized — enter your dashboard credentials",
        headers={"WWW-Authenticate": 'Basic realm="Trading Dashboard"'},
    )

# ── Retrain state ──────────────────────────────────────────────────────────

_retrain_running: bool = False
_retrain_log:     list[str] = []

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
    async def index(request: Request) -> Response:
        if not _check_auth(request):
            return _auth_required_response()
        # Serve page and set session cookie so JS API calls need no re-auth
        resp = HTMLResponse(_HTML)
        if DASHBOARD_USER and DASHBOARD_PASSWORD:
            resp.set_cookie(
                _COOKIE_NAME, _session_token(),
                max_age=86400 * 7,   # 7 days
                httponly=True,
                samesite="strict",
            )
        return resp

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/status")
    async def api_status(request: Request) -> Response:
        if not _check_auth(request):
            return _auth_required_response()
        return JSONResponse(_build_status(trader, agent))

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
                "reason":        t.exit_reason,
                "holding_hours": round(t.holding_hours, 1),
            }
            for t in reversed(trader.closed_trades[-50:])
        ]
        return {"trades": trades}

    @app.get("/api/equity")
    async def api_equity() -> dict:
        if trader is None:
            return {"curve": []}
        return {"curve": trader.equity_curve[-200:]}

    @app.get("/api/stats")
    async def api_stats() -> dict:
        """Comprehensive performance stats: overall + per-symbol + equity curve from DB."""
        overall: dict = {}
        by_symbol: list = []
        eq_curve: list = []

        if trader is not None:
            closed = trader.closed_trades
            total  = len(closed)
            wins   = [t for t in closed if t.pnl_usd > 0]
            losses = [t for t in closed if t.pnl_usd <= 0]
            gross_win  = sum(t.pnl_usd for t in wins)
            gross_loss = abs(sum(t.pnl_usd for t in losses))
            overall = {
                "total_trades":  total,
                "win_rate":      round(len(wins) / total * 100, 1) if total else 0.0,
                "total_pnl":     round(sum(t.pnl_usd for t in closed), 2),
                "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else 0.0,
                "avg_hold_h":    round(sum(t.holding_hours for t in closed) / total, 1) if total else 0.0,
            }
            # Per-symbol breakdown sorted by total PnL
            for sym in sorted(set(t.symbol for t in closed)):
                st = [t for t in closed if t.symbol == sym]
                sw = [t for t in st if t.pnl_usd > 0]
                by_symbol.append({
                    "symbol":   sym,
                    "trades":   len(st),
                    "wins":     len(sw),
                    "losses":   len(st) - len(sw),
                    "pnl_usd":  round(sum(t.pnl_usd for t in st), 2),
                    "win_rate": round(len(sw) / len(st) * 100, 1) if st else 0.0,
                    "avg_hold": round(sum(t.holding_hours for t in st) / len(st), 1) if st else 0.0,
                })
            by_symbol.sort(key=lambda x: x["pnl_usd"], reverse=True)

        # Equity curve from DB (works in both backtest and live mode)
        try:
            from src.utils.database import AsyncSessionFactory
            async with AsyncSessionFactory() as session:
                rows = (await session.execute(text("""
                    SELECT timestamp, total_value
                    FROM portfolio_snapshots
                    ORDER BY timestamp DESC LIMIT 400
                """))).fetchall()
                eq_curve = [
                    {"ts": r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]),
                     "equity": float(r[1])}
                    for r in reversed(rows)
                ]
        except Exception:
            pass

        return {"overall": overall, "by_symbol": by_symbol, "equity_curve": eq_curve}

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
    async def api_agent_enable(request: Request) -> Response:
        if not _check_auth(request):
            return _auth_required_response()
        if agent is None:
            return JSONResponse({"ok": False, "error": "Agent not initialized (check ANTHROPIC_API_KEY in .env)"})
        agent.enabled = True
        logger.info("AI agent ENABLED via dashboard")
        return JSONResponse({"ok": True, "enabled": True})

    @app.post("/api/agent/disable")
    async def api_agent_disable(request: Request) -> Response:
        if not _check_auth(request):
            return _auth_required_response()
        if agent is None:
            return JSONResponse({"ok": False, "error": "Agent not available"})
        agent.enabled = False
        logger.info("AI agent DISABLED via dashboard")
        return JSONResponse({"ok": True, "enabled": False})

    # ── Model retrain ──────────────────────────────────────────────────────

    @app.get("/api/retrain/status")
    async def api_retrain_status(request: Request) -> Response:
        if not _check_auth(request):
            return _auth_required_response()
        return JSONResponse({
            "running": _retrain_running,
            "log":     _retrain_log[-30:],
        })

    @app.post("/api/retrain")
    async def api_retrain(request: Request) -> Response:
        if not _check_auth(request):
            return _auth_required_response()
        global _retrain_running
        if _retrain_running:
            return JSONResponse({"ok": False, "error": "Retrain already in progress"})
        asyncio.create_task(_run_retrain())
        return JSONResponse({"ok": True, "message": "Retrain started in background"})

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


async def _run_retrain() -> None:
    """Run ML retraining in a subprocess so the trader keeps running."""
    global _retrain_running, _retrain_log
    _retrain_running = True
    _retrain_log = ["[retrain] Starting walk-forward training..."]
    logger.info("Model retrain started via dashboard")
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "main.py", "--mode", "train",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            _retrain_log.append(line)
            if len(_retrain_log) > 200:          # keep last 200 lines
                _retrain_log = _retrain_log[-200:]
        await proc.wait()
        _retrain_log.append(f"[retrain] Done — exit code {proc.returncode}")
        logger.info(f"Model retrain finished (exit {proc.returncode})")
    except Exception as exc:
        _retrain_log.append(f"[retrain] ERROR: {exc}")
        logger.error(f"Model retrain error: {exc}")
    finally:
        _retrain_running = False


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

  /* ── Chart ───────────────────────────────────────────────── */
  .chart-panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 20px; }
  .chart-wrap  { background: #0d1117; border: 1px solid #21262d; border-radius: 6px;
                 padding: 6px 8px; overflow: hidden; margin-top: 10px; }

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

  <!-- Performance stat cards -->
  <div class="cards">
    <div class="card"><label>Win Rate</label><div class="val" id="win-rate">—</div></div>
    <div class="card"><label>Total PnL (USDT)</label><div class="val" id="total-pnl">—</div></div>
    <div class="card"><label>Profit Factor</label><div class="val" id="profit-factor">—</div></div>
    <div class="card"><label>Avg Hold (hours)</label><div class="val gray" id="avg-hold">—</div></div>
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

  <!-- Model Retrain -->
  <div class="upload-panel">
    <div class="section-title">&#x1F9E0; ML Model Retrain</div>
    <div class="upload-desc">
      Retrain the LightGBM model on all data currently in the database.
      Runs in background — trading continues uninterrupted. Takes 5-15 minutes.
    </div>
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
      <button id="retrain-btn" onclick="startRetrain()"
        style="padding:8px 18px;background:#1f6feb;color:#fff;border:none;border-radius:6px;
               cursor:pointer;font-size:.83rem;font-weight:600">
        &#x25B6; Start Retrain
      </button>
      <span id="retrain-status" style="font-size:.8rem;color:#8b949e"></span>
    </div>
    <div id="retrain-log"
         style="display:none;margin-top:12px;background:#0d1117;border:1px solid #21262d;
                border-radius:6px;padding:10px 12px;font-size:.72rem;font-family:monospace;
                color:#8b949e;max-height:160px;overflow-y:auto;white-space:pre-wrap">
    </div>
  </div>

  <!-- Equity Curve Chart -->
  <div class="chart-panel">
    <div class="section-title">&#x1F4C8; Equity Curve</div>
    <div class="chart-wrap">
      <svg id="equity-chart" viewBox="0 0 800 140" preserveAspectRatio="none"
           style="width:100%;height:140px;display:block">
        <defs>
          <linearGradient id="eq-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#1f6feb" stop-opacity="0.25"/>
            <stop offset="100%" stop-color="#1f6feb" stop-opacity="0"/>
          </linearGradient>
        </defs>
        <path id="eq-fill" d="" fill="url(#eq-grad)"/>
        <polyline id="eq-line" points="" fill="none" stroke="#1f6feb" stroke-width="1.5" stroke-linejoin="round"/>
        <line id="eq-baseline" x1="0" y1="70" x2="800" y2="70"
              stroke="#30363d" stroke-width="1" stroke-dasharray="3,4"/>
        <text id="eq-empty" x="400" y="76" text-anchor="middle"
              fill="#8b949e" font-size="11" font-family="monospace">
          Waiting for data — portfolio snapshots appear here after first save
        </text>
      </svg>
    </div>
    <div id="chart-labels"
         style="display:flex;justify-content:space-between;font-size:.68rem;color:#8b949e;margin-top:4px;padding:0 4px">
    </div>
  </div>

  <!-- Per-symbol performance -->
  <div>
    <div class="section-title">&#x1F3AF; Performance by Symbol</div>
    <table>
      <thead><tr>
        <th>Symbol</th><th>Trades</th><th>Win Rate</th>
        <th>Total PnL</th><th>Wins</th><th>Losses</th><th>Avg Hold</th>
      </tr></thead>
      <tbody id="sym-body">
        <tr><td colspan="7" style="color:#8b949e;padding:14px">No closed trades yet</td></tr>
      </tbody>
    </table>
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
        <th>PnL USDT</th><th>PnL %</th><th>Hold</th><th>Reason</th><th>Closed</th>
      </tr></thead>
      <tbody id="trades-body">
        <tr><td colspan="9" style="color:#8b949e;padding:16px">Loading...</td></tr>
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

// ── Model retrain ─────────────────────────────────────────────────────────
let _retrainPoll = null;

async function startRetrain() {
  try {
    const r = await fetch('/api/retrain', { method: 'POST' });
    const d = await r.json();
    if (!d.ok) { alert('Retrain error: ' + (d.error || 'unknown')); return; }
    document.getElementById('retrain-btn').disabled = true;
    document.getElementById('retrain-log').style.display = 'block';
    document.getElementById('retrain-status').textContent = '⏳ Running...';
    document.getElementById('retrain-status').style.color = '#d29922';
    if (_retrainPoll) clearInterval(_retrainPoll);
    _retrainPoll = setInterval(pollRetrain, 3000);
  } catch(e) { alert('Request failed: ' + e); }
}

async function pollRetrain() {
  try {
    const r = await fetch('/api/retrain/status');
    const d = await r.json();
    const logEl = document.getElementById('retrain-log');
    logEl.textContent = (d.log || []).join('\n');
    logEl.scrollTop = logEl.scrollHeight;
    if (!d.running) {
      clearInterval(_retrainPoll); _retrainPoll = null;
      document.getElementById('retrain-btn').disabled = false;
      const done = (d.log || []).slice(-1)[0] || '';
      const ok = done.includes('exit code 0');
      document.getElementById('retrain-status').textContent = ok ? '✅ Done!' : '⚠ Finished (check log)';
      document.getElementById('retrain-status').style.color  = ok ? '#3fb950' : '#d29922';
    }
  } catch(e) {}
}

// ── Stats & equity chart ──────────────────────────────────────────────────
async function loadStats() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();
    const o = d.overall || {};

    // Performance cards
    setStatCard('win-rate',      o.win_rate,      v => fmt(v,1)+'%',  v => v >= 50 ? 'green' : 'red');
    setStatCard('total-pnl',     o.total_pnl,     v => (v>=0?'+':'')+fmt(v), v => v >= 0 ? 'green' : 'red');
    setStatCard('profit-factor', o.profit_factor, v => fmt(v,2),      v => v >= 1  ? 'green' : 'red');
    if (o.avg_hold_h != null) {
      const el = document.getElementById('avg-hold');
      if (el) { el.textContent = fmt(o.avg_hold_h,1)+'h'; el.className = 'val gray'; }
    }

    // Equity chart
    if (d.equity_curve && d.equity_curve.length > 1) drawEquityChart(d.equity_curve);

    // Per-symbol table
    const sb = document.getElementById('sym-body');
    if (!sb) return;
    if (!d.by_symbol || d.by_symbol.length === 0) return;
    sb.innerHTML = d.by_symbol.map(s => {
      const c = s.pnl_usd >= 0 ? 'green' : 'red';
      const sign = s.pnl_usd >= 0 ? '+' : '';
      return `<tr>
        <td><b>${s.symbol}</b></td>
        <td>${s.trades}</td>
        <td class="${s.win_rate >= 50 ? 'green' : 'red'}">${fmt(s.win_rate,1)}%</td>
        <td class="${c}">${sign}${fmt(s.pnl_usd)}</td>
        <td class="green">${s.wins}</td>
        <td class="red">${s.losses}</td>
        <td style="color:#8b949e">${fmt(s.avg_hold,1)}h</td>
      </tr>`;
    }).join('');
  } catch(e) { console.error('stats error:', e); }
}

function setStatCard(id, val, display, colorFn) {
  const el = document.getElementById(id);
  if (!el || val == null) return;
  el.textContent = display(val);
  el.className   = 'val ' + colorFn(val);
}

function drawEquityChart(points) {
  if (!points || points.length < 2) return;
  const W = 800, H = 140, pad = 8;
  const vals = points.map(p => p.equity);
  const mn = Math.min(...vals), mx = Math.max(...vals);
  const range = mx - mn || 1;
  const toX = i  => pad + (i / (points.length - 1)) * (W - 2 * pad);
  const toY = v  => pad + (1 - (v - mn) / range) * (H - 2 * pad);

  const ptStr = points.map((p,i) => `${toX(i).toFixed(1)},${toY(p.equity).toFixed(1)}`).join(' ');
  const fx = toX(0), lx = toX(points.length - 1);
  const fillD = `M${fx},${H} L${fx},${toY(points[0].equity).toFixed(1)} ` +
    points.map((p,i) => `L${toX(i).toFixed(1)},${toY(p.equity).toFixed(1)}`).join(' ') +
    ` L${lx},${H} Z`;

  document.getElementById('eq-line').setAttribute('points', ptStr);
  document.getElementById('eq-fill').setAttribute('d', fillD);
  const empty = document.getElementById('eq-empty');
  if (empty) empty.style.display = 'none';

  // Baseline at initial equity value
  const baseY = toY(points[0].equity).toFixed(1);
  const bl = document.getElementById('eq-baseline');
  if (bl) { bl.setAttribute('y1', baseY); bl.setAttribute('y2', baseY); }

  // Date labels (first / mid / last)
  const labels = document.getElementById('chart-labels');
  if (labels && points.length >= 2) {
    const d2 = dt => new Date(dt).toLocaleDateString('en-US',{month:'short',day:'2-digit'});
    const mid = points[Math.floor(points.length / 2)];
    labels.innerHTML =
      `<span>${d2(points[0].ts)}</span><span>${d2(mid.ts)}</span><span>${d2(points[points.length-1].ts)}</span>`;
  }
}

// ── Recent trades ──────────────────────────────────────────────────────────
async function loadTrades() {
  try {
    const r = await fetch('/api/trades');
    const d = await r.json();
    const tb = document.getElementById('trades-body');
    if (!d.trades || d.trades.length === 0) {
      tb.innerHTML = '<tr><td colspan="9" style="color:#8b949e;padding:14px">No closed trades yet</td></tr>';
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
        <td style="font-size:.72rem;color:#8b949e">${t.holding_hours ?? '—'}h</td>
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
loadStats();
loadDbStats();
setInterval(loadTrades,  15000);   // trades table every 15s
setInterval(loadStats,   30000);   // stats + chart every 30s
connect();
</script>
</body>
</html>
"""
