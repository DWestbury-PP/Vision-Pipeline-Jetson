#!/bin/bash

# ============================================================================
# Stop All Services - Clean Shutdown
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}🛑 Stopping All Services${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Stop native services using saved PIDs if available
echo -e "${YELLOW}Stopping native services...${NC}"

if [ -f logs/camera.pid ]; then
    PID=$(cat logs/camera.pid)
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID 2>/dev/null
        echo -e "  ${GREEN}✓${NC} Stopped Camera (PID: $PID)"
    fi
    rm logs/camera.pid
else
    pkill -f camera_native 2>/dev/null && echo -e "  ${GREEN}✓${NC} Stopped Camera"
fi

if [ -f logs/yolo.pid ]; then
    PID=$(cat logs/yolo.pid)
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID 2>/dev/null
        echo -e "  ${GREEN}✓${NC} Stopped YOLO (PID: $PID)"
    fi
    rm logs/yolo.pid
else
    pkill -f yolo_native 2>/dev/null && echo -e "  ${GREEN}✓${NC} Stopped YOLO"
fi

if [ -f logs/moondream.pid ]; then
    PID=$(cat logs/moondream.pid)
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID 2>/dev/null
        echo -e "  ${GREEN}✓${NC} Stopped Moondream (PID: $PID)"
    fi
    rm logs/moondream.pid
else
    pkill -f moondream_native 2>/dev/null && echo -e "  ${GREEN}✓${NC} Stopped Moondream"
fi

# Stop Docker services
echo ""
echo -e "${YELLOW}Stopping Docker services...${NC}"
docker-compose down

echo ""
echo -e "${GREEN}✅ All services stopped${NC}"
