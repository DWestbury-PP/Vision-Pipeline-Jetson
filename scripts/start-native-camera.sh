#!/bin/bash

# Start Native Camera service with Apple Studio Display access
echo "🚀 Starting Native Camera service..."

# Check if virtual environment exists
if [ ! -d "models/yolo11_env" ]; then
    echo "❌ Virtual environment not found at models/yolo11_env"
    echo "Please run the hybrid startup script first"
    exit 1
fi

# Activate virtual environment
source models/yolo11_env/bin/activate

# Check if Redis is running (try docker container first, then native)
if docker exec moondream-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis found in Docker container"
elif command -v redis-cli >/dev/null 2>&1 && redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis found running natively"
else
    echo "❌ Redis is not accessible. Please start Redis first:"
    echo "   docker-compose up redis -d"
    echo "   OR start Redis natively"
    exit 1
fi

echo "✅ Redis connection confirmed"
echo "✅ Virtual environment activated"

# Set environment variables
export REDIS_HOST=localhost
export REDIS_PORT=6379
export CAMERA_INDEX=0
export CAMERA_WIDTH=1280
export CAMERA_HEIGHT=720
export CAMERA_FPS=30
export CAMERA_FRAME_SKIP=1
export LOG_LEVEL=INFO

echo "🔧 Configuration:"
echo "   Camera Index: $CAMERA_INDEX (Apple Studio Display)"
echo "   Resolution: ${CAMERA_WIDTH}x${CAMERA_HEIGHT}"
echo "   Frame Rate: $CAMERA_FPS fps"
echo "   Frame Skip: $CAMERA_FRAME_SKIP (process every ${CAMERA_FRAME_SKIP}th frame)"
echo "   Redis: $REDIS_HOST:$REDIS_PORT"

# Start the service
echo "🎯 Starting Camera capture service..."
python3 services/native/camera_native.py
