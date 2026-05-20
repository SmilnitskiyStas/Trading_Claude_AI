#!/usr/bin/env bash
# Trading system VPS deployment script
# Ubuntu 24.04 LTS | 2 CPU | 4 GB RAM | 60 GB SSD
# Usage: bash deploy.sh

set -euo pipefail
BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${BOLD}=== $* ===${NC}"; }

# ── 1. Swap (important for 4GB RAM + Docker build) ─────────────────────────
section "Swap setup (4 GB)"
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    info "Swap 4 GB created"
else
    info "Swap already exists"
fi
free -h

# ── 2. System packages & Docker ────────────────────────────────────────────
section "System setup"
sudo apt-get update -qq
sudo apt-get install -y -qq curl git ufw htop

if ! command -v docker &>/dev/null; then
    info "Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    warn "Docker installed. Re-login required for group changes."
    warn "After re-login, run: bash deploy.sh again (it will skip this step)"
    exec sudo -u "$USER" bash "$0"   # re-exec as current user with docker group
else
    info "Docker: $(docker --version)"
fi

# ── 3. Clone from GitHub ───────────────────────────────────────────────────
section "Project setup"
PROJECT_DIR="${HOME}/trading"

if [ -d "${PROJECT_DIR}/.git" ]; then
    info "Repo exists — pulling latest..."
    cd "${PROJECT_DIR}" && git pull
else
    read -rp "Enter your GitHub repo URL (e.g. https://github.com/StasSmilnitskiy/trading-bot.git): " REPO_URL
    git clone "${REPO_URL}" "${PROJECT_DIR}"
    info "Cloned to ${PROJECT_DIR}"
fi

cd "${PROJECT_DIR}"
mkdir -p data/models logs

# ── 4. .env setup ─────────────────────────────────────────────────────────
section ".env configuration"
if [ ! -f .env ]; then
    cp .env.example .env
    warn ".env created from template. Edit it now:"
    warn "  nano ${PROJECT_DIR}/.env"
    echo ""
    echo "  Required fields to fill:"
    echo "    BINANCE_API_KEY, BINANCE_SECRET"
    echo "    BYBIT_API_KEY, BYBIT_SECRET"
    echo "    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID"
    echo "    POSTGRES_PASSWORD (set a strong password)"
    echo ""
    echo "  DATABASE_URL and REDIS_URL already correct for Docker (don't change them)"
    echo ""
    read -rp "Press Enter after editing .env..."
else
    info ".env already exists"
fi

# Verify .env has real values
if grep -q "your_binance_api_key\|your_telegram_bot_token" .env 2>/dev/null; then
    warn ".env still has placeholder values — make sure to fill real credentials"
fi

# ── 5. Firewall ────────────────────────────────────────────────────────────
section "Firewall (UFW)"
sudo ufw allow ssh
sudo ufw allow 8080/tcp
sudo ufw deny 5433/tcp   # PostgreSQL — internal only
sudo ufw deny 6379/tcp   # Redis — internal only
sudo ufw --force enable
info "Ports: 22 SSH + 8080 dashboard are open. DB and Redis are blocked externally."

# ── 6. Build Docker image ──────────────────────────────────────────────────
section "Docker image build"
info "Building image (5-15 min on first run — downloading PyTorch CPU ~600 MB)..."
docker compose build
info "Build complete"

# ── 7. Start infrastructure only (DB + Redis) ──────────────────────────────
section "Starting infrastructure"
docker compose up -d postgres redis
info "Waiting for PostgreSQL to be ready..."
sleep 15
docker compose exec postgres pg_isready -U trader -d trading && info "PostgreSQL OK" || warn "PostgreSQL not ready yet"

# ── 8. Download historical data ────────────────────────────────────────────
section "Downloading historical OHLCV data (2022–present)"
warn "This will take 10-30 minutes depending on exchange rate limits..."
docker compose run --rm \
    -e REDIS_URL=redis://redis:6379 \
    -e DATABASE_URL=postgresql://trader:${POSTGRES_PASSWORD:-trading_pass}@postgres:5432/trading \
    trader python main.py --mode download
info "Data download complete"

# ── 9. Start full system ───────────────────────────────────────────────────
section "Starting trading system"
docker compose up -d trader
info "Waiting 90s for startup..."
sleep 90

if curl -sf "http://localhost:8080/health" >/dev/null 2>&1; then
    info "Dashboard UP"
else
    warn "Dashboard not ready yet — check: docker compose logs -f trader"
fi

docker compose ps

# ── 10. Summary ────────────────────────────────────────────────────────────
section "Deployment complete!"

# Get public IP
PUBLIC_IP=$(curl -sf https://api.ipify.org 2>/dev/null || echo "YOUR_VPS_IP")

echo ""
echo -e "  ${BOLD}Dashboard:${NC}  http://${PUBLIC_IP}:8080"
echo ""
echo "  Useful commands:"
echo "    docker compose logs -f trader      # live logs"
echo "    docker compose logs -f --tail 50   # last 50 lines all services"
echo "    docker compose restart trader      # restart after code update"
echo "    docker compose down                # stop everything"
echo "    docker compose exec postgres psql -U trader trading  # DB shell"
echo ""
echo "  Update system from GitHub:"
echo "    cd ~/trading && git pull && docker compose build && docker compose up -d"
echo ""
