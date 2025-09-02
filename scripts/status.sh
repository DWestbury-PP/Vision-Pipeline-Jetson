#!/bin/bash

# ============================================================================
# System Status Monitor
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}📊 Moondream Vision Pipeline Status${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check Docker services
echo -e "${YELLOW}🐳 Docker Services:${NC}"
docker-compose ps --format "table {{.Service}}\t{{.Status}}" 2>/dev/null | head -6 || echo "  Docker services not running"

echo ""
echo -e "${YELLOW}🖥️  Native Services:${NC}"

# Check Camera
if pgrep -f camera_native > /dev/null; then
    FPS=$(tail -1 logs/camera_native.log 2>/dev/null | grep -oE '[0-9]+\.[0-9]+ fps' | head -1)
    echo -e "  🎥 Camera:    ${GREEN}Running${NC} ${FPS}"
else
    echo -e "  🎥 Camera:    ${RED}Stopped${NC}"
fi

# Check YOLO
if pgrep -f yolo_native > /dev/null; then
    FPS=$(tail -5 logs/yolo_native.log 2>/dev/null | grep -oE 'fps: [0-9]+\.[0-9]+' | tail -1)
    echo -e "  🎯 YOLO:      ${GREEN}Running${NC} ${FPS}"
else
    echo -e "  🎯 YOLO:      ${RED}Stopped${NC}"
fi

# Check Moondream
if pgrep -f moondream_native > /dev/null; then
    echo -e "  🌙 Moondream: ${GREEN}Running${NC}"
else
    echo -e "  🌙 Moondream: ${RED}Stopped${NC}"
fi

echo ""
echo -e "${YELLOW}📡 Redis Channels:${NC}"
if docker exec moondream-redis redis-cli ping > /dev/null 2>&1; then
    echo "  Camera frames subscribers: $(docker exec moondream-redis redis-cli --raw PUBSUB NUMSUB frame:camera.frames | tail -1)"
    echo "  YOLO results subscribers:  $(docker exec moondream-redis redis-cli --raw PUBSUB NUMSUB msg:detection.yolo | tail -1)"
    echo "  Chat requests subscribers: $(docker exec moondream-redis redis-cli --raw PUBSUB NUMSUB msg:chat.requests | tail -1)"
else
    echo "  Redis not accessible"
fi

echo ""
echo -e "${YELLOW}🌐 Endpoints:${NC}"
echo -e "  Frontend:  ${GREEN}http://localhost:3000${NC}"
echo -e "  API Docs:  ${GREEN}http://localhost:8000/docs${NC}"

echo ""
echo -e "${BLUE}============================================${NC}"
