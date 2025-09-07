#!/bin/bash

# ============================================================================
# Vision Pipeline Mac - Complete Stack Startup
# ============================================================================
# This script starts all services in the correct order for the hybrid architecture
# - Docker services: Redis, API, Frontend, Fusion
# - Native services: Camera, YOLO, Moondream (for Apple Silicon GPU access)
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}🚀 Vision Pipeline Mac Startup${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Check Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"

# Check virtual environment
if [ ! -d "models/yolo11_env" ]; then
    echo -e "${RED}❌ Virtual environment not found at models/yolo11_env${NC}"
    echo "Please run the setup script first"
    exit 1
fi

echo -e "${GREEN}✅ Virtual environment found${NC}"

# Check if base image exists, build if not
if ! docker images | grep -q "moondream-base"; then
    echo -e "${YELLOW}📦 Building base image (first time setup)...${NC}"
    if [ -f "scripts/build-base.sh" ]; then
        ./scripts/build-base.sh
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Failed to build base image${NC}"
            exit 1
        fi
        echo -e "${GREEN}✅ Base image built${NC}"
    else
        echo -e "${RED}❌ Build script not found: scripts/build-base.sh${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Base image exists${NC}"
fi

# Create logs directory
mkdir -p logs

# ============================================================================
# STEP 1: Stop any existing services
# ============================================================================
echo ""
echo -e "${YELLOW}🛑 Stopping any existing services...${NC}"

# Stop native services
pkill -f camera_native 2>/dev/null || true
pkill -f yolo_native 2>/dev/null || true
pkill -f moondream_native 2>/dev/null || true

# Stop Docker services
docker-compose down 2>/dev/null || true

sleep 2

# ============================================================================
# STEP 2: Start Docker services
# ============================================================================
echo ""
echo -e "${BLUE}🐳 Starting Docker services...${NC}"

# Start Redis first (required by all other services)
echo -e "  Starting Redis..."
DOCKER_BUILDKIT=0 docker-compose up -d redis
sleep 3

# Verify Redis is healthy
if docker exec vision-pipeline-redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}  ✅ Redis is healthy${NC}"
else
    echo -e "${RED}  ❌ Redis failed to start${NC}"
    exit 1
fi

# Start API and Fusion services
echo -e "  Starting API service..."
DOCKER_BUILDKIT=0 docker-compose up -d api
sleep 3

echo -e "  Starting Fusion service..."
DOCKER_BUILDKIT=0 docker-compose up -d fusion
sleep 2

# Start Frontend
echo -e "  Starting Frontend..."
DOCKER_BUILDKIT=0 docker-compose up -d frontend
sleep 2

# Verify Docker services
echo ""
echo -e "${YELLOW}📊 Docker services status:${NC}"
docker-compose ps --format "table {{.Service}}\t{{.Status}}" | head -6

# ============================================================================
# STEP 3: Start Native Services (in order)
# ============================================================================
echo ""
echo -e "${BLUE}🖥️  Starting native services (Apple Silicon GPU)...${NC}"

# Activate virtual environment for all native services
source models/yolo11_env/bin/activate

# Set common environment variables
export REDIS_HOST=localhost
export REDIS_PORT=6379
export LOG_LEVEL=INFO

# Start Camera Service
echo -e "${YELLOW}  🎥 Starting Camera service...${NC}"
export CAMERA_INDEX=0
export CAMERA_WIDTH=1920
export CAMERA_HEIGHT=1080
export CAMERA_FPS=6  # Optimized for processing pipeline
export CAMERA_FRAME_SKIP=1

# Add a small delay to ensure camera is ready
sleep 2
python3 services/native/camera_native.py > logs/camera_native.log 2>&1 &
CAMERA_PID=$!
sleep 5

# Verify Camera started
if ps -p $CAMERA_PID > /dev/null; then
    echo -e "${GREEN}    ✅ Camera service started (PID: $CAMERA_PID)${NC}"
else
    echo -e "${RED}    ❌ Camera service failed to start${NC}"
    echo "    Check logs/camera_native.log for errors"
    exit 1
fi

# Start YOLO Service
echo -e "${YELLOW}  🎯 Starting YOLO service...${NC}"
export YOLO_MODEL=yolo11n.pt
export YOLO_DEVICE=mps
export YOLO_CONFIDENCE=0.5
export YOLO_FRAME_STRIDE=2

python3 services/native/yolo_native.py > logs/yolo_native.log 2>&1 &
YOLO_PID=$!
sleep 5  # YOLO needs time to load model

# Verify YOLO started
if ps -p $YOLO_PID > /dev/null; then
    echo -e "${GREEN}    ✅ YOLO service started (PID: $YOLO_PID)${NC}"
else
    echo -e "${RED}    ❌ YOLO service failed to start${NC}"
    echo "    Check logs/yolo_native.log for errors"
    exit 1
fi

# Start Moondream Service
echo -e "${YELLOW}  🌙 Starting Moondream service...${NC}"
export VLM_FRAME_STRIDE=10
export VLM_MAX_CONTEXT_LENGTH=100

python3 services/native/moondream_native.py > logs/moondream_native.log 2>&1 &
MOONDREAM_PID=$!
sleep 3

# Verify Moondream started
if ps -p $MOONDREAM_PID > /dev/null; then
    echo -e "${GREEN}    ✅ Moondream service started (PID: $MOONDREAM_PID)${NC}"
else
    echo -e "${YELLOW}    ⚠️  Moondream service may have issues${NC}"
    echo "    Check logs/moondream_native.log for errors"
    # Don't exit - system can work without Moondream
fi

# ============================================================================
# STEP 4: Final Status
# ============================================================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}🎉 System Started Successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${BLUE}📊 Service Status:${NC}"
echo -e "  🐳 Docker Services:"
echo -e "    • Redis:    ${GREEN}Running${NC}"
echo -e "    • API:      ${GREEN}Running${NC}"
echo -e "    • Frontend: ${GREEN}Running${NC}"
echo -e "    • Fusion:   ${GREEN}Running${NC}"
echo ""
echo -e "  🖥️  Native Services (PIDs):"
echo -e "    • Camera:    ${GREEN}$CAMERA_PID${NC}"
echo -e "    • YOLO:      ${GREEN}$YOLO_PID${NC}"
echo -e "    • Moondream: ${GREEN}$MOONDREAM_PID${NC}"
echo ""
echo -e "${BLUE}🌐 Access Points:${NC}"
echo -e "  • Frontend:  ${GREEN}http://localhost:3000${NC}"
echo -e "  • API Docs:  ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo -e "${BLUE}📝 Management Commands:${NC}"
echo -e "  • View logs:       tail -f logs/<service>_native.log"
echo -e "  • Docker logs:     docker-compose logs <service> --follow"
echo -e "  • Stop all:        ./scripts/stop-all.sh"
echo -e "  • Monitor status:  ./scripts/status.sh"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo -e "  • Camera publishes at ~6 FPS"
echo -e "  • YOLO processes every 2nd frame"
echo -e "  • Moondream processes every 10th frame"
echo -e "  • Enable detection overlays in the UI"
echo ""

# Save PIDs for stop script
echo "$CAMERA_PID" > logs/camera.pid
echo "$YOLO_PID" > logs/yolo.pid
echo "$MOONDREAM_PID" > logs/moondream.pid

echo -e "${GREEN}✨ Enjoy your computer vision pipeline!${NC}"
