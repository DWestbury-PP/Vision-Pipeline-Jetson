#!/bin/bash

# Start Native YOLO11 service with Apple Silicon GPU support
echo "🚀 Starting Native YOLO11 service..."

# Check if virtual environment exists
if [ ! -d "models/yolo11_env" ]; then
    echo "❌ Virtual environment not found at models/yolo11_env"
    echo "Please run: python3 -m venv models/yolo11_env"
    echo "Then: source models/yolo11_env/bin/activate && pip install ultralytics redis pydantic pydantic-settings"
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
export YOLO_MODEL=yolo11n.pt
export YOLO_DEVICE=mps  # Apple Silicon
export YOLO_CONFIDENCE=0.5
export LOG_LEVEL=INFO

echo "🔧 Configuration:"
echo "   Model: $YOLO_MODEL"
echo "   Device: $YOLO_DEVICE (Apple Silicon MPS)"
echo "   Confidence: $YOLO_CONFIDENCE"
echo "   Redis: $REDIS_HOST:$REDIS_PORT"

# Start the service
echo "🎯 Starting YOLO11 inference service..."
python3 services/native/yolo_native.py
