#!/bin/bash

# Start Native Moondream service using existing CLI
echo "🚀 Starting Native Moondream service..."

# Check if moondream CLI is available
if ! command -v moondream &> /dev/null; then
    echo "❌ Moondream CLI not found. Please install it first:"
    echo "   Visit: https://github.com/vikhyat/moondream"
    echo "   Or ensure 'moondream' is in your PATH"
    exit 1
fi

echo "✅ Moondream CLI found: $(which moondream)"

# Test moondream CLI
if ! moondream --help > /dev/null 2>&1; then
    echo "❌ Moondream CLI test failed"
    exit 1
fi

echo "✅ Moondream CLI is working"

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

# Set environment variables
export REDIS_HOST=localhost
export REDIS_PORT=6379
export VLM_FRAME_STRIDE=10
export MOONDREAM_CLI_COMMAND=moondream
export LOG_LEVEL=INFO

echo "🔧 Configuration:"
echo "   CLI Command: $MOONDREAM_CLI_COMMAND"
echo "   Frame Stride: $VLM_FRAME_STRIDE (process every ${VLM_FRAME_STRIDE}th frame)"
echo "   Redis: $REDIS_HOST:$REDIS_PORT"

# Activate virtual environment for dependencies
source models/yolo11_env/bin/activate

# Start the service
echo "🎯 Starting Moondream VLM service..."
python3 services/native/moondream_native.py
