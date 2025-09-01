#!/bin/bash

# Moondream Vision Pipeline Deployment Script
set -e

echo "🚀 Deploying Moondream Vision Pipeline"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose >/dev/null 2>&1; then
    echo "❌ docker-compose is not installed. Please install docker-compose and try again."
    exit 1
fi

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file from env.example"
    cp env.example .env
fi

# Pull Redis image
echo "📦 Pulling Redis image..."
docker pull redis:7-alpine

# Build all services
echo "🔨 Building all services..."

echo "  📷 Building camera service..."
docker-compose build camera

echo "  🎯 Building YOLO service..."
docker-compose build yolo

echo "  🌙 Building Moondream service..."
docker-compose build moondream

echo "  🔗 Building fusion service..."
docker-compose build fusion

echo "  🌐 Building API service..."
docker-compose build api

echo "  💻 Building frontend..."
docker-compose build frontend

# Start services
echo "🏃 Starting all services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
docker-compose ps

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "📡 API: http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"
echo ""
echo "To view logs: docker-compose logs -f [service_name]"
echo "To stop: docker-compose down"
echo "To rebuild: docker-compose build"
