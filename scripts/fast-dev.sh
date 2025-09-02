#!/bin/bash

# Fast development script using optimized base images
echo "🚀 Starting fast development environment..."

# Disable BuildKit to use legacy builder (which properly resolves local images)
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

# Check if base image exists
if ! docker images | grep -q "moondream-base"; then
    echo "❌ Base image not found. Please run ./scripts/build-base.sh first"
    exit 1
fi

echo "✅ Base image found"

# Use legacy builder for fast builds with base images
echo "🔧 Building services with fast builds..."

# Stop any existing services
docker-compose down 2>/dev/null || true

# Build and start services using the main docker-compose.yml
docker-compose up "$@"
