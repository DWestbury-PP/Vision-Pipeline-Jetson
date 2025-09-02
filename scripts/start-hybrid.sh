#!/bin/bash

# Hybrid Architecture Startup Script
# Starts containerized services + native GPU services

echo "🌟 Starting Moondream Hybrid Architecture"
echo "   📦 Containers: Redis, API, Frontend, Fusion"
echo "   🖥️  Native: YOLO11 + Moondream (Apple Silicon GPU)"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker not found. Please install Docker Desktop."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ docker-compose not found. Please install Docker Compose."
    exit 1
fi

if ! command_exists python3; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Start containerized services (excluding camera, yolo, moondream)
echo ""
echo "📦 Starting containerized services..."
echo "   🔴 Redis (message bus)"
echo "   🔴 API (FastAPI backend)"  
echo "   🔴 Frontend (React UI)"
echo "   🔴 Fusion (result combiner)"

./scripts/compose.sh up redis api frontend fusion --build -d

if [ $? -ne 0 ]; then
    echo "❌ Failed to start containerized services"
    exit 1
fi

echo "✅ Containerized services started"

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check if Redis is accessible
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not accessible. Check container logs:"
    echo "   docker-compose logs redis"
    exit 1
fi

echo "✅ Redis is ready"

# Check if API is accessible
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  API health check failed, but continuing..."
    echo "   Check API logs: docker-compose logs api"
fi

# Start native services in background
echo ""
echo "🖥️  Starting native GPU services..."

# Start YOLO11 service
echo "   🎯 Starting YOLO11 (Apple Silicon MPS)..."
./scripts/start-native-yolo.sh &
YOLO_PID=$!
sleep 2

# Start Moondream service
echo "   🌙 Starting Moondream (native CLI)..."
./scripts/start-native-moondream.sh &
MOONDREAM_PID=$!
sleep 2

echo ""
echo "🎉 Hybrid architecture started successfully!"
echo ""
echo "📊 Service Status:"
echo "   🔗 Frontend:    http://localhost:3000"
echo "   🔗 API:         http://localhost:8000"
echo "   🔗 API Docs:    http://localhost:8000/docs"
echo "   🔗 WebSocket:   ws://localhost:8001"
echo ""
echo "🖥️  Native Services (PIDs):"
echo "   🎯 YOLO11:      $YOLO_PID"
echo "   🌙 Moondream:   $MOONDREAM_PID"
echo ""
echo "📋 Management Commands:"
echo "   View container logs:  docker-compose logs [service] --follow"
echo "   Stop containers:      docker-compose down"
echo "   Kill native services: kill $YOLO_PID $MOONDREAM_PID"
echo ""
echo "⚡ Performance Benefits:"
echo "   • Apple Silicon GPU acceleration for ML models"
echo "   • Containerized message bus and API for reliability"
echo "   • Best of both worlds: native performance + container isolation"
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait for user interrupt
trap 'echo ""; echo "🛑 Shutting down hybrid architecture..."; kill $YOLO_PID $MOONDREAM_PID 2>/dev/null; docker-compose down; echo "✅ Shutdown complete"; exit 0' INT

# Keep script running
wait
