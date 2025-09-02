#!/bin/bash

# Build optimized base image for fast development iterations
echo "🏗️  Building optimized base image..."

# Build the base image (this will take time but only needs to be done once)
docker build -f containers/Dockerfile.base-optimized -t moondream-base:latest .

if [ $? -eq 0 ]; then
    echo "✅ Base image built successfully!"
    echo "📦 Image size:"
    docker images moondream-base:latest
    echo ""
    echo "🚀 You can now use fast builds with:"
    echo "   ./scripts/fast-dev.sh"
    echo ""
    echo "   Or directly with docker-compose:"
    echo "   docker-compose up --build"
    echo ""
    echo "⚡ Fast rebuilds will now take seconds instead of minutes!"
else
    echo "❌ Base image build failed!"
    exit 1
fi
