# ПРОМПТ ТЗ — Автоматична ML/AI Торгова Система (Crypto Trading Bot)

## Контекст проєкту

Розробити повноцінну автоматичну торгову систему для криптовалютного ринку з використанням ML моделей, AI агента та технічного аналізу. Система запускається на сервері у безперервному режимі. Перший етап — Paper Trading (симуляція без реальних грошей) для перевірки якості моделей.

**Біржі**: Binance, Bybit, Kraken, OKX — через уніфікований інтерфейс `ccxt`
**Монети**: Top 10 за капіталізацією (BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, TRX, LINK)
**Моніторинг**: Telegram бот + веб дашборд
**Середовище розробки**: Python 3.11+, локальний ноутбук i7-14700, 32GB RAM, без дискретної GPU (Intel integrated only)

---

## Архітектура системи

```
trading-system/
├── data/                        # Зберігання даних
│   ├── raw/                     # Сирі OHLCV дані (по біржах)
│   │   ├── binance/
│   │   ├── bybit/
│   │   ├── kraken/
│   │   └── okx/
│   ├── processed/               # Оброблені фічі
│   └── models/                  # Збережені ML моделі (.pkl)
├── src/
│   ├── exchanges/               # Абстракція бірж
│   │   ├── base_exchange.py     # Базовий клас (інтерфейс)
│   │   ├── binance_exchange.py  # Binance специфіка
│   │   ├── bybit_exchange.py    # Bybit специфіка
│   │   ├── kraken_exchange.py   # Kraken специфіка
│   │   ├── okx_exchange.py      # OKX специфіка
│   │   └── exchange_factory.py  # Фабрика — вибір біржі
│   ├── data_pipeline/           # Збір та обробка даних
│   │   ├── collector.py         # Завантаження OHLCV з усіх бірж
│   │   ├── aggregator.py        # Агрегація даних між біржами
│   │   ├── processor.py         # Feature engineering
│   │   └── news_fetcher.py      # Новини та sentiment
│   ├── models/                  # ML моделі
│   │   ├── lgbm_model.py        # LightGBM (основна модель)
│   │   ├── lstm_model.py        # LSTM (опціонально)
│   │   └── trainer.py           # Навчання та walk-forward валідація
│   ├── strategy/                # Торгові стратегії
│   │   ├── signals.py           # Генерація сигналів
│   │   ├── risk_manager.py      # Управління ризиками
│   │   ├── portfolio.py         # Управління портфелем
│   │   └── arbitrage.py         # Міжбіржовий арбітраж (майбутнє)
│   ├── trading/                 # Виконання угод
│   │   ├── paper_trader.py      # Paper trading симулятор
│   │   ├── order_router.py      # Вибір найкращої біржі для угоди
│   │   └── live_trader.py       # Live trading (майбутнє)
│   ├── ai_agent/                # AI агент
│   │   ├── agent.py             # Claude API агент
│   │   └── news_analyzer.py     # Аналіз новин через LLM
│   ├── monitoring/              # Моніторинг
│   │   ├── telegram_bot.py      # Telegram сповіщення
│   │   ├── dashboard.py         # Веб дашборд (FastAPI)
│   │   └── metrics.py           # Розрахунок метрик
│   └── utils/
│       ├── config.py            # Конфігурація
│       ├── logger.py            # Логування
│       ├── database.py          # SQLite / PostgreSQL
│       └── cache.py             # Redis клієнт та хелпери
├── notebooks/                   # Jupyter для експериментів
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── tests/
├── .env
├── .env.example
├── .gitignore
├── .claudeignore
├── requirements.txt
├── docker-compose.yml
└── main.py
```

### .claudeignore (обов'язково створити першим):
```
data/
logs/
notebooks/
__pycache__/
.venv/
*.pyc
*.pkl
*.h5
*.csv
*.parquet
*.log
*.db
*.sqlite
dump.rdb
```

---

## Ключова концепція — Мульти-біржева абстракція

Всі біржі працюють через єдиний інтерфейс. Це дозволяє:
- Порівнювати ціни між біржами (найкраща ціна виконання)
- Збирати більше даних для навчання ML
- У майбутньому — міжбіржовий арбітраж

```python
# base_exchange.py — єдиний інтерфейс для всіх бірж
from abc import ABC, abstractmethod

class BaseExchange(ABC):

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list:
        """Завантажити OHLCV свічки"""
        pass

    @abstractmethod
    def fetch_ticker(self, symbol: str) -> dict:
        """Поточна ціна"""
        pass

    @abstractmethod
    def fetch_order_book(self, symbol: str, limit: int) -> dict:
        """Стакан ордерів"""
        pass

    @abstractmethod
    def create_order(self, symbol: str, side: str, amount: float, price: float):
        """Створити ордер (для live trading)"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def rate_limit(self) -> int:
        """Запитів на хвилину"""
        pass


# exchange_factory.py — вибір біржі
import ccxt

class ExchangeFactory:
    EXCHANGES = {
        'binance': {'class': ccxt.binance, 'rate_limit': 1200},
        'bybit':   {'class': ccxt.bybit,   'rate_limit': 600},
        'kraken':  {'class': ccxt.kraken,  'rate_limit': 60},
        'okx':     {'class': ccxt.okx,     'rate_limit': 300},
    }

    @classmethod
    def create(cls, exchange_name: str, api_key: str = None, secret: str = None):
        config = cls.EXCHANGES[exchange_name]
        return config['class']({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })

    @classmethod
    def create_all(cls, credentials: dict) -> dict:
        """Повертає словник всіх підключених бірж"""
        return {
            name: cls.create(name, **creds)
            for name, creds in credentials.items()
        }
```

---

## Redis — кешування та черги

Redis виконує 3 ролі в системі: кеш для швидких даних, pub/sub для передачі сигналів між модулями, та rate-limit лічильник для захисту від перевищення лімітів бірж.

```python
# utils/cache.py
import redis.asyncio as aioredis
import json
from typing import Any, Optional

class RedisCache:
    def __init__(self, url: str = "redis://localhost:6379"):
        self.client = aioredis.from_url(url, decode_responses=True)

    # --- Поточні ціни (TTL 5 секунд) ---
    async def set_ticker(self, exchange: str, symbol: str, data: dict):
        key = f"ticker:{exchange}:{symbol}"
        await self.client.setex(key, 5, json.dumps(data))

    async def get_ticker(self, exchange: str, symbol: str) -> Optional[dict]:
        key = f"ticker:{exchange}:{symbol}"
        data = await self.client.get(key)
        return json.loads(data) if data else None

    # --- ML сигнали (TTL 1 година) ---
    async def set_signal(self, symbol: str, signal: dict):
        key = f"signal:{symbol}"
        await self.client.setex(key, 3600, json.dumps(signal))

    async def get_signal(self, symbol: str) -> Optional[dict]:
        key = f"signal:{symbol}"
        data = await self.client.get(key)
        return json.loads(data) if data else None

    # --- AI агент відповідь (TTL 1 година, економія токенів) ---
    async def set_agent_response(self, response: dict):
        await self.client.setex("agent:market_analysis", 3600, json.dumps(response))

    async def get_agent_response(self) -> Optional[dict]:
        data = await self.client.get("agent:market_analysis")
        return json.loads(data) if data else None

    # --- Sentiment новин (TTL 30 хвилин) ---
    async def set_sentiment(self, symbol: str, score: float):
        key = f"sentiment:{symbol}"
        await self.client.setex(key, 1800, str(score))

    async def get_sentiment(self, symbol: str) -> Optional[float]:
        data = await self.client.get(key=f"sentiment:{symbol}")
        return float(data) if data else None

    # --- Поточний стан портфеля (TTL 10 секунд, для дашборду) ---
    async def set_portfolio_snapshot(self, data: dict):
        await self.client.setex("portfolio:live", 10, json.dumps(data))

    async def get_portfolio_snapshot(self) -> Optional[dict]:
        data = await self.client.get("portfolio:live")
        return json.loads(data) if data else None

    # --- Pub/Sub: публікація нового сигналу ---
    async def publish_signal(self, symbol: str, signal: dict):
        await self.client.publish(f"signals:{symbol}", json.dumps(signal))

    # --- Rate limit лічильник для бірж ---
    async def check_rate_limit(self, exchange: str, max_per_minute: int) -> bool:
        key = f"ratelimit:{exchange}"
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)
        results = await pipe.execute()
        current_count = results[0]
        return current_count <= max_per_minute

    # --- Стан підключення бірж ---
    async def set_exchange_status(self, exchange: str, is_online: bool):
        key = f"exchange:status:{exchange}"
        await self.client.setex(key, 30, "1" if is_online else "0")

    async def get_exchange_status(self, exchange: str) -> Optional[bool]:
        data = await self.client.get(f"exchange:status:{exchange}")
        return bool(int(data)) if data else None
```

### Що і навіщо кешується в Redis:

| Ключ | TTL | Навіщо |
|------|-----|--------|
| `ticker:{exchange}:{symbol}` | 5 сек | Поточна ціна без зайвих API запитів |
| `signal:{symbol}` | 1 год | ML сигнал не перераховується кожну секунду |
| `agent:market_analysis` | 1 год | Claude API викликається раз на годину, не на кожен тік |
| `sentiment:{symbol}` | 30 хв | FinBERT inference — повільна операція |
| `portfolio:live` | 10 сек | Дашборд читає з Redis, не з БД |
| `ratelimit:{exchange}` | 60 сек | Лічильник запитів щоб не перевищити ліміт біржі |
| `exchange:status:{exchange}` | 30 сек | Heartbeat: чи жива біржа |

### Pub/Sub — передача сигналів між модулями:

```python
# Коли ML модель генерує новий сигнал → публікує в Redis
# Paper trader підписується і отримує миттєво

# publisher (signals.py)
await cache.publish_signal("BTC/USDT", {
    "action": "buy",
    "confidence": 0.74,
    "price": 67432.0,
    "timestamp": "2024-01-15T14:00:00Z"
})

# subscriber (paper_trader.py)
async def listen_signals():
    pubsub = cache.client.pubsub()
    await pubsub.psubscribe("signals:*")
    async for message in pubsub.listen():
        if message["type"] == "pmessage":
            signal = json.loads(message["data"])
            await paper_trader.execute_signal(signal)
```

---

## Rate Limits — важливо для паралельних запитів

| Біржа   | Запитів/хв | Затримка між запитами | Особливості |
|---------|-----------|----------------------|-------------|
| Binance | 1200      | ~50ms                | найбільша ліквідність |
| Bybit   | 600       | ~100ms               | хороші ф'ючерси |
| Kraken  | 60        | ~1000ms              | найнижчий ліміт! |
| OKX     | 300       | ~200ms               | хороший API |

```python
# collector.py — враховувати різні rate limits
import asyncio

async def collect_all_exchanges(symbols, timeframe, since):
    tasks = []
    for exchange_name, exchange in exchanges.items():
        # Kraken потребує більших затримок
        delay = 1.0 if exchange_name == 'kraken' else 0.1
        tasks.append(
            collect_with_delay(exchange, symbols, timeframe, since, delay)
        )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return aggregate_results(results)
```

---

## ФАЗА 1 — Data Pipeline

### 1.1 Збір OHLCV з усіх бірж

```python
# Для кожної монети збираємо дані з ВСІХ бірж
# Таймфрейми: 1h (основний), 4h, 1d
# Глибина: 3 роки (2022-01-01 → сьогодні)

# Пріоритет джерел даних:
# - Binance: основне джерело (найбільше даних, висока ліквідність)
# - Bybit / OKX: додаткові джерела, верифікація
# - Kraken: для BTC/ETH (найдавніша історія з 2013р)

# Агрегація між біржами:
# - Для навчання ML: використовувати середньозважену ціну (VWAP across exchanges)
# - Зберігати окремо дані кожної біржі для порівняння спредів
```

### 1.2 Додаткові дані специфічні для бірж

```python
# Bybit / Binance — funding rate (для ф'ючерсів)
# Показує настрій ринку: позитивний = лонги домінують
funding_rate = exchange.fetch_funding_rate(symbol)

# Всі біржі — order book imbalance
# Співвідношення bid/ask показує тиск купівлі/продажу
order_book = exchange.fetch_order_book(symbol, limit=20)
bid_volume = sum([b[1] for b in order_book['bids']])
ask_volume = sum([a[1] for a in order_book['asks']])
imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

# Ці фічі додати до feature engineering
extra_features = ['funding_rate', 'order_book_imbalance', 'spread_bps']
```

### 1.3 Feature Engineering

```python
technical_features = [
    # Trend
    'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'sma_20', 'sma_50',
    'macd', 'macd_signal', 'macd_hist',
    'adx_14',                          # сила тренду

    # Momentum
    'rsi_14', 'rsi_7',
    'stoch_k', 'stoch_d',
    'cci_20',
    'williams_r',
    'roc_10',                          # Rate of Change

    # Volatility
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
    'atr_14',
    'atr_percent',                     # ATR / close (нормалізований)
    'keltner_upper', 'keltner_lower',

    # Volume
    'volume_sma_20',
    'volume_ratio',
    'obv',
    'vwap',
    'cmf_20',                          # Chaikin Money Flow

    # Cross-exchange
    'binance_bybit_spread',            # спред між біржами
    'order_book_imbalance',
    'funding_rate',                    # якщо доступно

    # Price action
    'price_change_1h', 'price_change_4h', 'price_change_24h',
    'high_low_ratio',
    'close_position',

    # Seasonality
    'hour_of_day', 'day_of_week', 'day_of_month', 'month',
    'is_weekend',

    # BTC dominance (для альткоїнів)
    'btc_price_change_1h',             # BTC як ринковий індикатор
]

# Target
# 1  = Buy  (ціна зросте >1% за наступні 4 свічки)
# -1 = Sell (ціна впаде  >1% за наступні 4 свічки)
# 0  = Hold
```

### 1.4 Sentiment аналіз новин

```python
# Джерела:
# - CryptoPanic API (безкоштовний tier, 100 req/day)
# - RSS: CoinDesk, CoinTelegraph, Decrypt
# - (опціонально) Twitter/X API для реального часу

# NLP модель:
# "ElKulako/cryptobert" — навчена спеціально на крипто новинах
# Або "ProsusAI/finbert" — загальна фінансова модель

# Результат на вихід:
sentiment_output = {
    'symbol': 'BTC',
    'sentiment_score': 0.73,      # -1.0 до +1.0
    'sentiment_24h_avg': 0.41,
    'news_count_24h': 47,
    'top_headlines': [...],
}
```

---

## ФАЗА 2 — ML Модель (LightGBM)

### 2.1 LightGBM класифікатор (оптимізований для CPU)

```python
import lightgbm as lgb

params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'n_jobs': -1,               # всі 20 ядер i7-14700
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'early_stopping_rounds': 50,
    'class_weight': 'balanced', # важливо: класи нерівномірні
    'verbose': -1,
}

# Optuna для підбору гіперпараметрів (запускати раз)
# Результати зберігати і перевикористовувати
```

### 2.2 Walk-Forward валідація (обов'язково!)

```
НЕ використовувати звичайний train_test_split — це data leakage!

Walk-Forward схема (кожен fold зсувається на 1 місяць):
[=====Train 12m=====][Val 2m][                    ]
[  =====Train 12m=====][Val 2m][                  ]
[    =====Train 12m=====][Val 2m][                ]

Fold розміри:
- Train: 12 місяців
- Validation: 2 місяці
- Gap: 1 тиждень (щоб уникнути leakage)
- Всього folds: ~12 (на 3 роках даних)
```

### 2.3 Метрики

```python
metrics_to_track = {
    'accuracy', 'precision_buy', 'recall_buy',
    'f1_macro', 'roc_auc_ovr',
    'feature_importance_top20',
    'confusion_matrix',
}
# Зберігати результати кожного fold окремо
# Фінальна оцінка = середнє по всіх folds
```

---

## ФАЗА 3 — AI Агент (Claude API)

```python
# agent.py
# Модель: claude-sonnet-4-20250514
# Виклик: раз на годину (економія токенів)
# Кешувати відповідь між викликами

# Задачі агента:
# 1. Аналіз останніх новин → market_sentiment: bullish/bearish/neutral
# 2. Виявлення екстремальних подій:
#    - Регуляторні заборони → STOP trading
#    - Злам біржі → STOP trading на цій біржі
#    - Халвінг / великі анонси → підвищити вагу сигналів
# 3. Щоденний звіт о 09:00 UTC → відправити в Telegram
# 4. Корекція ML сигналів: якщо агент = bearish AND ML = Buy → Hold

# Промпт для агента (system):
AGENT_SYSTEM_PROMPT = """
Ти — AI аналітик криптовалютного ринку.
Отримуєш: список останніх новин + поточні ціни + метрики портфеля.
Повертаєш JSON:
{
  "market_sentiment": "bullish|bearish|neutral",
  "sentiment_score": float,  // -1.0 до 1.0
  "risk_level": "low|medium|high|extreme",
  "trading_allowed": bool,
  "key_events": [...],
  "recommendation": "string",
  "affected_symbols": {
    "BTC": {"sentiment": float, "reason": "string"},
    ...
  }
}
"""
```

---

## ФАЗА 4 — Risk Management

```python
class RiskManager:
    # Позиції
    MAX_POSITION_SIZE = 0.10       # 10% портфеля на 1 монету
    MAX_TOTAL_EXPOSURE = 0.80      # 80% в позиціях, 20% в кеші
    MAX_CORRELATED_POSITIONS = 3   # не >3 монет з кореляцією >0.8

    # Захист
    STOP_LOSS_PERCENT = 0.03       # -3% стоп-лос
    TAKE_PROFIT_PERCENT = 0.06     # +6% тейк-профіт (R:R = 1:2)
    TRAILING_STOP = 0.02           # трейлінг стоп 2%

    # Денні ліміти
    MAX_DAILY_LOSS = 0.05          # зупинити якщо -5% за день
    MAX_TRADES_PER_DAY = 20        # не більше 20 угод/день
    MAX_TRADES_PER_SYMBOL = 3      # не більше 3 угод/день на монету

    # Вибір біржі для виконання
    def select_best_exchange(self, symbol, side, amount):
        """Вибрати біржу з найкращою ціною та ліквідністю"""
        prices = {ex: ex.fetch_ticker(symbol) for ex in self.exchanges}
        if side == 'buy':
            return min(prices, key=lambda x: prices[x]['ask'])
        return max(prices, key=lambda x: prices[x]['bid'])

    def kelly_position_size(self, win_rate, avg_win, avg_loss):
        """Kelly Criterion для оптимального розміру позиції"""
        if avg_loss == 0:
            return 0
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return min(kelly * 0.5, self.MAX_POSITION_SIZE)  # Half-Kelly (безпечніше)

    def check_drawdown(self, current_value, peak_value):
        """Зупинити торгівлю якщо просадка > 15%"""
        drawdown = (peak_value - current_value) / peak_value
        if drawdown > 0.15:
            self.halt_trading("Max drawdown exceeded")
```

---

## ФАЗА 5 — Paper Trading симулятор

```python
class PaperTrader:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance        # USDT
        self.positions = {}                   # {symbol: {qty, entry_price, exchange}}
        self.trade_history = []
        self.peak_value = initial_balance
        self.start_time = datetime.now()

    def execute_signal(self, signal):
        """
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'buy|sell|hold',
            'confidence': 0.73,
            'ml_signal': 1,
            'agent_sentiment': 'bullish',
            'best_exchange': 'binance',
            'price': 67432.0,
        }
        """
        ...

    def get_metrics(self) -> dict:
        """Розрахувати всі метрики"""
        returns = self._calculate_returns()
        return {
            'total_return_pct': ...,
            'sharpe_ratio': ...,
            'sortino_ratio': ...,
            'max_drawdown': ...,
            'win_rate': ...,
            'profit_factor': ...,
            'total_trades': ...,
            'avg_trade_duration_hours': ...,
            'best_trade': ...,
            'worst_trade': ...,
            'by_exchange': {...},      # метрики по кожній біржі
            'by_symbol': {...},        # метрики по кожній монеті
        }
```

---

## ФАЗА 6 — Моніторинг

### 6.1 Telegram Bot

```
Команди:
/status          — стан системи, баланс, P&L
/positions       — відкриті позиції з P&L
/stats [period]  — статистика (today/week/month/all)
/trades [n]      — останні N угод (default 10)
/exchanges       — статус підключення всіх бірж
/model           — метрики ML моделі
/stop            — EMERGENCY STOP (зупинити всі угоди)
/resume          — відновити торгівлю
/report          — повний звіт PDF

Автоматичні сповіщення:
📈 Нова покупка: BTC/USDT @ $67,432 (Binance) | Розмір: $1,000
📉 Продаж: ETH/USDT @ $3,241 | P&L: +$87.3 (+2.8%)
🛑 Стоп-лос: SOL/USDT | Збиток: -$43.2 (-3.0%)
⚠️  Drawdown перевищив 10%! Поточний: -11.3%
🔴 ПОМИЛКА: Kraken API недоступний
📊 Денний звіт [00:00 UTC]: +$234 (+2.3%) | 12 угод | Sharpe: 1.8
```

### 6.2 Веб Дашборд (FastAPI + HTML/JS)

```
GET /                  — Overview dashboard
GET /api/portfolio     — JSON: баланс, позиції, метрики
GET /api/trades        — JSON: історія угод з фільтрами
GET /api/signals       — JSON: останні сигнали ML + агента
GET /api/exchanges     — JSON: статус і метрики бірж
GET /api/model/stats   — JSON: метрики ML моделі
GET /api/news          — JSON: останні новини + sentiment
WS  /ws/live           — WebSocket: live оновлення цін і P&L

Дашборд відображає:
- Portfolio equity curve (графік)
- P&L по монетах (bar chart)
- Win rate, Sharpe, Drawdown (live)
- Таблиця відкритих позицій
- Стрічка останніх угод
- Статус всіх 4 бірж (online/offline/rate-limited)
- Sentiment gauge (AI агент)
```

---

## База даних

```sql
-- SQLite для розробки → PostgreSQL для продакшну

CREATE TABLE ohlcv_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    timestamp BIGINT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    UNIQUE(exchange, symbol, timeframe, timestamp)
);

CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20),
    exchange VARCHAR(20),          -- на якій біржі виконано
    side VARCHAR(10),              -- buy/sell
    entry_price REAL,
    exit_price REAL,
    quantity REAL,
    pnl REAL,
    pnl_percent REAL,
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    exit_reason VARCHAR(50),       -- take_profit/stop_loss/signal/manual
    ml_confidence REAL,            -- впевненість ML моделі при вході
    agent_sentiment VARCHAR(20),   -- bullish/bearish/neutral
    is_paper BOOLEAN DEFAULT 1
);

CREATE TABLE portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_value REAL,
    cash_balance REAL,
    positions_value REAL,
    daily_pnl REAL,
    total_pnl REAL,
    drawdown REAL,
    sharpe_ratio REAL
);

CREATE TABLE exchange_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange VARCHAR(20),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_online BOOLEAN,
    latency_ms INTEGER,
    error_message TEXT
);

CREATE TABLE news_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20),
    timestamp TIMESTAMP,
    headline TEXT,
    sentiment_score REAL,
    source VARCHAR(100),
    url TEXT
);

CREATE TABLE ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP,
    symbol VARCHAR(20),
    exchange VARCHAR(20),
    signal INTEGER,               -- 1=Buy, 0=Hold, -1=Sell
    confidence REAL,
    features_json TEXT            -- знімок фіч для дебагу
);
```

---

## Redis — кешування та черги

Redis виконує 3 ролі в системі: кеш API відповідей, pub/sub шина між сервісами, та тимчасове сховище стану.

### Що зберігаємо в Redis і чому

```python
# cache.py
import redis.asyncio as redis
import json
from typing import Any, Optional

class CacheManager:
    def __init__(self, url: str):
        self.client = redis.from_url(url, decode_responses=True)

    # --- 1. ЦІНИ (TTL: 10 секунд) ---
    # Щоб не смикати біржі кожну секунду від кожного модуля
    async def cache_ticker(self, exchange: str, symbol: str, data: dict):
        key = f"ticker:{exchange}:{symbol}"
        await self.client.setex(key, 10, json.dumps(data))

    async def get_ticker(self, exchange: str, symbol: str) -> Optional[dict]:
        data = await self.client.get(f"ticker:{exchange}:{symbol}")
        return json.loads(data) if data else None

    # --- 2. AI АГЕНТ (TTL: 1 година) ---
    # Claude API коштує токени — кешуємо відповідь агента на 60 хвилин
    async def cache_agent_analysis(self, analysis: dict):
        await self.client.setex("agent:analysis", 3600, json.dumps(analysis))

    async def get_agent_analysis(self) -> Optional[dict]:
        data = await self.client.get("agent:analysis")
        return json.loads(data) if data else None

    # --- 3. SENTIMENT НОВИН (TTL: 30 хвилин) ---
    # FinBERT inference важкий — не запускати для кожного запиту
    async def cache_sentiment(self, symbol: str, score: float):
        key = f"sentiment:{symbol}"
        await self.client.setex(key, 1800, str(score))

    async def get_sentiment(self, symbol: str) -> Optional[float]:
        data = await self.client.get(f"sentiment:{symbol}")
        return float(data) if data else None

    # --- 4. ML СИГНАЛИ (TTL: 5 хвилин) ---
    # Модель не треба гнати на кожен тік — достатньо раз на свічку
    async def cache_signal(self, symbol: str, signal: dict):
        key = f"signal:{symbol}"
        await self.client.setex(key, 300, json.dumps(signal))

    async def get_signal(self, symbol: str) -> Optional[dict]:
        data = await self.client.get(f"signal:{symbol}")
        return json.loads(data) if data else None

    # --- 5. СТАН PAPER TRADER (без TTL) ---
    # Портфель завжди в Redis — швидкий доступ з дашборду і Telegram
    async def set_portfolio_state(self, state: dict):
        await self.client.set("portfolio:state", json.dumps(state))

    async def get_portfolio_state(self) -> Optional[dict]:
        data = await self.client.get("portfolio:state")
        return json.loads(data) if data else None

    # --- 6. PUB/SUB — повідомлення між сервісами ---
    # Trader публікує → Telegram бот і дашборд отримують
    async def publish_trade(self, trade: dict):
        await self.client.publish("channel:trades", json.dumps(trade))

    async def publish_alert(self, alert: dict):
        await self.client.publish("channel:alerts", json.dumps(alert))

    # --- 7. RATE LIMIT ЛІЧИЛЬНИКИ (TTL: 60 секунд) ---
    # Відстежувати скільки запитів зробили до кожної біржі
    async def increment_rate_limit(self, exchange: str) -> int:
        key = f"ratelimit:{exchange}"
        count = await self.client.incr(key)
        if count == 1:
            await self.client.expire(key, 60)
        return count

    async def check_rate_limit(self, exchange: str, max_per_minute: int) -> bool:
        count = await self.client.get(f"ratelimit:{exchange}")
        return (int(count) if count else 0) < max_per_minute
```

### Карта TTL для всіх ключів

```
ticker:{exchange}:{symbol}   → 10 сек    (ціни)
signal:{symbol}              → 5 хв      (ML сигнали)
sentiment:{symbol}           → 30 хв     (NLP результат)
agent:analysis               → 60 хв     (Claude API відповідь)
news:headlines               → 15 хв     (список новин)
portfolio:state              → без TTL   (поточний стан портфеля)
ratelimit:{exchange}         → 60 сек    (лічильник запитів)
```

### Pub/Sub канали

```
channel:trades    → нова угода виконана
channel:alerts    → важливе сповіщення (stop-loss, drawdown)
channel:prices    → live ціни для WebSocket дашборду
channel:signals   → новий ML сигнал
```

---

## Docker — повний setup

### Що запускається в Docker, що локально

```
В Docker (завжди):        Redis, PostgreSQL (продакшн)
В Docker або локально:    Trading app, Dashboard, Telegram bot
Локально ТІЛЬКИ:          Jupyter notebooks, навчання ML моделі
```

### docker-compose.yml

```yaml
version: '3.9'

services:

  # ─── ІНФРАСТРУКТУРА ───────────────────────────────────────────

  redis:
    image: redis:7-alpine
    container_name: trading_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: >
      redis-server
      --appendonly yes
      --appendfsync everysec
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  postgres:
    image: postgres:16-alpine
    container_name: trading_postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: trading
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./src/utils/schema.sql:/docker-entrypoint-initdb.d/schema.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trader -d trading"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ─── ЗАСТОСУНОК ───────────────────────────────────────────────

  trader:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: trading_bot
    restart: unless-stopped
    env_file: .env
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://trader:${POSTGRES_PASSWORD}@postgres:5432/trading
    volumes:
      - ./data:/app/data          # OHLCV дані та ML моделі
      - ./logs:/app/logs
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    command: python main.py --mode paper_trade
    deploy:
      resources:
        limits:
          memory: 8G              # достатньо для LightGBM inference
          cpus: '4'

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: trading_dashboard
    restart: unless-stopped
    env_file: .env
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://trader:${POSTGRES_PASSWORD}@postgres:5432/trading
    ports:
      - "8080:8080"
    depends_on:
      - redis
      - postgres
    command: python main.py --mode dashboard

  # ─── МОНІТОРИНГ (опціонально) ─────────────────────────────────

  redis_commander:
    image: rediscommander/redis-commander:latest
    container_name: trading_redis_ui
    restart: unless-stopped
    environment:
      - REDIS_HOSTS=local:redis:6379
    ports:
      - "8081:8081"
    depends_on:
      - redis
    profiles:
      - debug                    # запускається тільки з --profile debug

volumes:
  redis_data:
  postgres_data:
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Системні залежності для LightGBM та pandas-ta
RUN apt-get update && apt-get install -y \
    gcc g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Спочатку копіюємо тільки requirements (кеш шарів Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Потім решту коду
COPY src/ ./src/
COPY main.py .
COPY .env.example .

# Директорії для даних (монтуються як volumes)
RUN mkdir -p data/raw data/processed data/models logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
```

### Команди Docker

```bash
# ─── РОЗРОБКА (тільки інфраструктура в Docker) ───

# Запустити Redis + PostgreSQL
docker-compose up redis postgres -d

# Перевірити що Redis живий
docker exec trading_redis redis-cli ping
# → PONG

# Переглянути ключі в Redis (під час розробки)
docker exec -it trading_redis redis-cli
> KEYS *
> GET portfolio:state
> TTL agent:analysis

# ─── ПОВНИЙ ЗАПУСК ───

# Зібрати образи і запустити все
docker-compose up --build -d

# Переглянути логи
docker-compose logs -f trader
docker-compose logs -f dashboard

# Перезапустити тільки бота (після зміни коду)
docker-compose restart trader

# ─── DEBUG РЕЖИМ (з Redis Commander UI) ───

docker-compose --profile debug up -d
# Redis Commander доступний на http://localhost:8081

# ─── ЗУПИНКА ───

# М'яка зупинка
docker-compose stop

# Повне видалення (БЕЗ видалення даних)
docker-compose down

# Повне видалення З видаленням даних (обережно!)
docker-compose down -v

# ─── НАВЧАННЯ МОДЕЛІ (локально, не в Docker) ───
# Навчання запускати ЛОКАЛЬНО — Docker не потрібен для цього
python main.py --mode train
```

### .dockerignore

```
data/raw/
data/processed/
logs/
notebooks/
.venv/
__pycache__/
*.pyc
*.pkl
*.parquet
*.csv
.git/
.env
*.log
```

### Схема взаємодії сервісів

```
┌─────────────────────────────────────────────────────┐
│                   Docker Network                    │
│                                                     │
│  ┌──────────┐    pub/sub    ┌─────────────────────┐ │
│  │  trader  │ ────────────▶ │       Redis         │ │
│  │  (bot)   │ ◀──────────── │  cache + pub/sub    │ │
│  └──────────┘               └──────────┬──────────┘ │
│       │                                │             │
│       │ SQL                            │ subscribe   │
│       ▼                                ▼             │
│  ┌──────────┐               ┌─────────────────────┐ │
│  │ postgres │               │     dashboard       │ │
│  │   (db)   │◀──────────────│   (FastAPI+WS)      │ │
│  └──────────┘    SQL read   └──────────┬──────────┘ │
│                                        │             │
└────────────────────────────────────────┼─────────────┘
                                         │ HTTP/WS
                                    ┌────▼────┐
                                    │ Browser │
                                    │localhost│
                                    │  :8080  │
                                    └─────────┘
```

---

## Конфігурація (.env)

```env
# === БІРЖІ ===
# Binance
BINANCE_API_KEY=your_key
BINANCE_SECRET=your_secret

# Bybit
BYBIT_API_KEY=your_key
BYBIT_SECRET=your_secret

# Kraken
KRAKEN_API_KEY=your_key
KRAKEN_SECRET=your_secret

# OKX
OKX_API_KEY=your_key
OKX_SECRET=your_secret
OKX_PASSPHRASE=your_passphrase

# === АКТИВНІ БІРЖІ (comma-separated) ===
ACTIVE_EXCHANGES=binance,bybit,kraken,okx
PRIMARY_EXCHANGE=binance         # для paper trading симуляції

# === НОВИНИ ===
CRYPTOPANIC_API_KEY=your_key

# === AI АГЕНТ ===
ANTHROPIC_API_KEY=your_key

# === TELEGRAM ===
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# === БАЗА ДАНИХ ===
DATABASE_URL=sqlite:///data/trading.db

# === REDIS ===
REDIS_URL=redis://localhost:6379
REDIS_DB=0

# === ТОРГІВЛЯ ===
PAPER_TRADING=true
INITIAL_BALANCE=10000
MAX_POSITION_SIZE=0.10
STOP_LOSS=0.03
TAKE_PROFIT=0.06
MAX_DAILY_LOSS=0.05

# === МОНІТОРИНГ ===
DASHBOARD_PORT=8080
DASHBOARD_HOST=0.0.0.0
LOG_LEVEL=INFO
```

---

## Покроковий план реалізації

### ✅ Крок 1 — Фундамент (1-2 дні) → перший запуск
1. Структура проєкту + `.claudeignore` (перший файл!)
2. `requirements.txt`
3. `config.py` + `.env` + `.env.example`
4. `logger.py` (loguru)
5. `database.py` + schema SQL
6. `cache.py` — Redis клієнт з перевіркою з'єднання
7. `docker-compose.yml` — запустити Redis + PostgreSQL: `docker-compose up redis postgres -d`
8. **Перевірка**: `python main.py --mode check` — Redis ping OK, DB connection OK

### ✅ Крок 2 — Підключення бірж (1-2 дні) → перше з'єднання
1. `base_exchange.py` — абстрактний клас
2. `exchange_factory.py` — фабрика через ccxt
3. `binance_exchange.py`, `bybit_exchange.py`, `kraken_exchange.py`, `okx_exchange.py`
4. **Перевірка**: тест підключення до кожної біржі, fetch BTC ціни

### ✅ Крок 3 — Дані (2-3 дні) → перші дані в БД
1. `collector.py` — завантаження 3 років OHLCV з усіх бірж
2. `aggregator.py` — агрегація між біржами
3. `processor.py` — всі технічні індикатори
4. **Перевірка**: notebook `01_data_exploration.ipynb`

### ✅ Крок 4 — ML модель (3-5 днів) → перші сигнали
1. `lgbm_model.py` — LightGBM класифікатор
2. `trainer.py` — walk-forward валідація
3. `signals.py` — генерація сигналів
4. **Перевірка**: notebook `03_model_training.ipynb` — метрики моделі

### ✅ Крок 5 — Paper Trading (2-3 дні) → перший "прибуток"
1. `risk_manager.py` — Kelly Criterion, ліміти
2. `paper_trader.py` — симулятор
3. `metrics.py` — Sharpe, Drawdown тощо
4. **Перевірка**: backtest на 6 місяцях даних → звіт у консоль

### ✅ Крок 6 — Моніторинг (2-3 дні) → видно що відбувається
1. `telegram_bot.py` — /status, /positions, /trades
2. `dashboard.py` — FastAPI + HTML дашборд
3. **Перевірка**: відкрити http://localhost:8080, перевірити Telegram команди

### ✅ Крок 7 — AI Агент (1-2 дні) → розумні рішення
1. `news_fetcher.py` — CryptoPanic + RSS
2. `news_analyzer.py` — FinBERT sentiment
3. `agent.py` — Claude API інтеграція
4. **Перевірка**: агент аналізує новини і повертає JSON

### ✅ Крок 8 — Інтеграція (1-2 дні) → повний цикл
1. `main.py` — головний цикл + scheduler
2. `docker-compose.yml`
3. End-to-end тест: дані → модель → сигнал → paper trade → Telegram → дашборд
4. **Перевірка**: система працює 24 години без помилок

---

## Залежності (requirements.txt)

```
# Trading & Exchanges
ccxt==4.3.0
pandas==2.2.0
numpy==1.26.0
pandas-ta==0.3.14b0

# ML
lightgbm==4.3.0
scikit-learn==1.4.0
optuna==3.5.0
joblib==1.3.2

# NLP / Sentiment
transformers==4.37.0
torch==2.2.0
feedparser==6.0.11

# API & Web
fastapi==0.109.0
uvicorn==0.27.0
websockets==12.0
httpx==0.26.0
python-telegram-bot==21.0

# Database
sqlalchemy==2.0.25
aiosqlite==0.20.0
asyncpg==0.29.0            # PostgreSQL async (продакшн)

# Cache
redis==5.0.1               # Redis клієнт з asyncio підтримкою

# Utils
python-dotenv==1.0.0
loguru==0.7.2
tqdm==4.66.1
apscheduler==3.10.4
anthropic==0.18.0

# Visualization (notebooks)
plotly==5.18.0
matplotlib==3.8.0
jupyter==1.0.0
```

---

## Очікуваний результат після Кроку 5

```
=== BACKTEST RESULTS (2023-01-01 → 2023-12-31) ===
Exchanges: Binance, Bybit, Kraken, OKX
Symbols:   BTC ETH BNB SOL XRP DOGE ADA AVAX TRX LINK
Initial:   $10,000 USDT

Portfolio Value:    $13,240 USDT
Total Return:       +32.4%
Sharpe Ratio:       1.89
Sortino Ratio:      2.41
Max Drawdown:       -8.7%
Win Rate:           55.1%
Profit Factor:      1.53
Total Trades:       312

Best Exchange:      Binance (+34.1% виконань)
Best Symbol:        SOL/USDT (+67.3%)
Worst Symbol:       DOGE/USDT (-4.2%)

Model Accuracy:     58.3% (walk-forward avg)
Avg Confidence:     0.67
```

---

## Важливі зауваження для Claude Code

1. **Перший файл** — `.claudeignore`, щоб не читати data/ та logs/
2. **Порядок** — строго Кроки 1→8, не пропускати
3. **Kraken rate limit** — мінімум 1 секунда між запитами, інакше бан
4. **Paper trading = true** завжди під час розробки, перевіряти в кожній функції
5. **ccxt уніфікує API** — не писати окремий код для кожної біржі де можна уникнути
6. **Try/except + retry** на всіх мережевих викликах (exponential backoff)
7. **Логувати все**: кожен сигнал, угоду, помилку API, рішення агента
8. **SQLite зараз**, але код писати через SQLAlchemy ORM для легкої міграції на PostgreSQL
9. **Walk-forward валідація** — обов'язково, звичайний split = data leakage = фальшиві результати
10. **Тести** — мінімум для: collector, processor, risk_manager, paper_trader
11. **Redis перевіряти при старті** — якщо Redis недоступний, система все одно стартує (деградований режим без кешу), але логує попередження
12. **Навчання ML моделі — ЛОКАЛЬНО**, не в Docker: потребує багато RAM і CPU, Docker overhead не потрібен
13. **Docker volumes** — data/ і logs/ монтувати як volumes, щоб дані не зникали при перезапуску контейнера
14. **Не хардкодити** REDIS_URL і DATABASE_URL — читати з env, щоб легко перемикатись між локальним і Docker

---

## Команди запуску

```bash
# Перевірка конфігурації
python main.py --mode check

# Завантаження даних (всі біржі)
python main.py --mode download --exchanges binance,bybit,kraken,okx

# Тільки одна біржа (для швидкого старту)
python main.py --mode download --exchanges binance

# Навчання моделі
python main.py --mode train

# Backtest на збережених даних
python main.py --mode backtest --from 2023-01-01 --to 2023-12-31

# Paper trading (live режим)
python main.py --mode paper_trade

# Тільки дашборд
python main.py --mode dashboard

# Все разом (production режим)
python main.py --mode all
```
