#!/bin/bash

# ============================================================================
# Quick Start - Minimal startup for development
# ============================================================================

set -e

echo "🚀 Quick Start - Moondream Vision Pipeline"
echo ""

# Create logs directory
mkdir -p logs

# Stop existing services
pkill -f camera_native 2>/dev/null || true
pkill -f yolo_native 2>/dev/null || true
pkill -f moondream_native 2>/dev/null || true
docker-compose down 2>/dev/null || true

echo "Starting services..."

# Start Docker services
DOCKER_BUILDKIT=0 docker-compose up -d redis api frontend

# Wait for Redis
sleep 3

# Start native services
source models/yolo11_env/bin/activate
export REDIS_HOST=localhost
export CAMERA_FPS=6

python3 services/native/camera_native.py > logs/camera_native.log 2>&1 &
sleep 3

python3 services/native/yolo_native.py > logs/yolo_native.log 2>&1 &
sleep 3

python3 services/native/moondream_native.py > logs/moondream_native.log 2>&1 &

echo ""
echo "✅ Services started!"
echo "🌐 Frontend: http://localhost:3000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Run ./scripts/stop-all.sh to stop all services"
