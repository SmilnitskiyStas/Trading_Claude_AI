-- PostgreSQL schema (used by docker-entrypoint-initdb.d)

CREATE TABLE IF NOT EXISTS ohlcv_data (
    id        SERIAL PRIMARY KEY,
    exchange  VARCHAR(20) NOT NULL,
    symbol    VARCHAR(20) NOT NULL,
    timeframe VARCHAR(5)  NOT NULL,
    timestamp BIGINT      NOT NULL,
    open      DOUBLE PRECISION,
    high      DOUBLE PRECISION,
    low       DOUBLE PRECISION,
    close     DOUBLE PRECISION,
    volume    DOUBLE PRECISION,
    UNIQUE (exchange, symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf ON ohlcv_data (symbol, timeframe, timestamp DESC);

CREATE TABLE IF NOT EXISTS trades (
    id               SERIAL PRIMARY KEY,
    symbol           VARCHAR(20),
    exchange         VARCHAR(20),
    side             VARCHAR(10),
    entry_price      DOUBLE PRECISION,
    exit_price       DOUBLE PRECISION,
    quantity         DOUBLE PRECISION,
    pnl              DOUBLE PRECISION,
    pnl_percent      DOUBLE PRECISION,
    entry_time       TIMESTAMPTZ,
    exit_time        TIMESTAMPTZ,
    exit_reason      VARCHAR(50),
    ml_confidence    DOUBLE PRECISION,
    agent_sentiment  VARCHAR(20),
    is_paper         BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, entry_time DESC);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    total_value     DOUBLE PRECISION,
    cash_balance    DOUBLE PRECISION,
    positions_value DOUBLE PRECISION,
    daily_pnl       DOUBLE PRECISION,
    total_pnl       DOUBLE PRECISION,
    drawdown        DOUBLE PRECISION,
    sharpe_ratio    DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS exchange_status (
    id            SERIAL PRIMARY KEY,
    exchange      VARCHAR(20),
    timestamp     TIMESTAMPTZ DEFAULT NOW(),
    is_online     BOOLEAN,
    latency_ms    INTEGER,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS news_sentiment (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20),
    timestamp       TIMESTAMPTZ,
    headline        TEXT,
    sentiment_score DOUBLE PRECISION,
    source          VARCHAR(100),
    url             TEXT
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id            SERIAL PRIMARY KEY,
    timestamp     TIMESTAMPTZ,
    symbol        VARCHAR(20),
    exchange      VARCHAR(20),
    signal        SMALLINT,
    confidence    DOUBLE PRECISION,
    features_json TEXT
);
