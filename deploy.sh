#!/usr/bin/env bash
# Trading system VPS deployment script
# Tested on Ubuntu 22.04 / 24.04
# Usage: bash deploy.sh

set -euo pipefail
BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
section() { echo -e "\n${BOLD}=== $* ===${NC}"; }

# ── 1. System update & Docker ──────────────────────────────────────────────
section "System setup"
sudo apt-get update -qq
sudo apt-get install -y -qq curl git ufw

if ! command -v docker &>/dev/null; then
    info "Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    info "Docker installed. You may need to re-login for group changes."
else
    info "Docker already installed: $(docker --version)"
fi

# ── 2. Project directory ───────────────────────────────────────────────────
section "Project directory"
PROJECT_DIR="${HOME}/trading"
mkdir -p "${PROJECT_DIR}/data/models" "${PROJECT_DIR}/logs"
info "Project dir: ${PROJECT_DIR}"

# ── 3. Copy files from local machine ──────────────────────────────────────
section "File sync"
warn "On your LOCAL Windows machine run these commands to upload files:"
echo ""
echo "  # Upload source code (from project root):"
echo "  scp -r src main.py requirements.txt Dockerfile docker-compose.yml .env.example \\"
echo "      user@YOUR_VPS_IP:~/trading/"
echo ""
echo "  # Upload .env with your real credentials:"
echo "  scp .env user@YOUR_VPS_IP:~/trading/.env"
echo ""
echo "  # Upload the trained ML model:"
echo "  scp data/models/lgbm_final.pkl user@YOUR_VPS_IP:~/trading/data/models/"
echo ""
read -rp "Press Enter when files are uploaded..."

# ── 4. Verify required files ───────────────────────────────────────────────
section "Checking required files"
cd "${PROJECT_DIR}"

MISSING=0
for f in main.py src/utils/config.py .env data/models/lgbm_final.pkl docker-compose.yml; do
    if [[ -f "$f" ]]; then
        info "OK  $f"
    else
        echo -e "${RED}[MISS]${NC} $f"
        MISSING=1
    fi
done

if [[ $MISSING -eq 1 ]]; then
    echo -e "${RED}Some files are missing. Upload them and re-run.${NC}"
    exit 1
fi

# ── 5. Firewall ────────────────────────────────────────────────────────────
section "Firewall (UFW)"
sudo ufw allow ssh
sudo ufw allow 8080/tcp   # dashboard
sudo ufw --force enable
info "Ports open: 22 (SSH), 8080 (dashboard)"
warn "PostgreSQL (5433) and Redis (6379) are NOT exposed externally — internal Docker only."

# ── 6. Build & start ───────────────────────────────────────────────────────
section "Docker build & start"
cd "${PROJECT_DIR}"

info "Building Docker image (first time takes 5-10 min)..."
docker compose build --no-cache

info "Starting services..."
docker compose up -d

# ── 7. Health check ────────────────────────────────────────────────────────
section "Health check"
info "Waiting 90 seconds for startup..."
sleep 90

if curl -sf "http://localhost:8080/health" >/dev/null; then
    info "Dashboard is UP: http://localhost:8080"
else
    warn "Dashboard not responding yet. Check logs:"
    echo "  docker compose logs -f trader"
fi

docker compose ps

# ── 8. Useful commands ─────────────────────────────────────────────────────
section "Useful commands"
echo "  View logs:       docker compose logs -f trader"
echo "  Stop system:     docker compose down"
echo "  Restart trader:  docker compose restart trader"
echo "  Open dashboard:  http://YOUR_VPS_IP:8080"
echo "  DB shell:        docker compose exec postgres psql -U trader trading"
echo "  Redis CLI:       docker compose exec redis redis-cli"
echo ""
info "Deployment complete!"
