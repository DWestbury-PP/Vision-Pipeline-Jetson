#!/bin/bash

# Setup Models for Vision Pipeline
# This script ensures required models are available for the containerized services

echo "🚀 Setting up models for Vision Pipeline..."

# Create models directory structure
mkdir -p models/yolo
mkdir -p models/moondream

echo "📁 Created model directories"

# Check if YOLO model exists
if [ ! -f "models/yolo/yolo11n.pt" ]; then
    echo "⬇️  Downloading YOLO11n model..."
    cd models/yolo
    
    # Download YOLO11n model (lightweight version)
    if command -v wget >/dev/null 2>&1; then
        wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt
    else
        echo "❌ Neither wget nor curl found. Please install one of them or download manually:"
        echo "   https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt"
        echo "   Save to: models/yolo/yolo11n.pt"
    fi
    
    cd ../..
else
    echo "✅ YOLO model already exists"
fi

# Check if Moondream model exists
if [ ! -d "models/moondream/moondream2" ] || [ ! -f "models/moondream/moondream2/model.safetensors" ]; then
    echo "⬇️  Setting up Moondream model..."
    
    # Check if git-lfs is available
    if command -v git-lfs >/dev/null 2>&1; then
        echo "📦 Using git-lfs to download Moondream model..."
        cd models/moondream
        
        # Clone the model repository
        if [ ! -d "moondream2" ]; then
            git clone https://huggingface.co/vikhyatk/moondream2
        else
            echo "📁 Moondream repository already exists, pulling latest..."
            cd moondream2
            git pull
            cd ..
        fi
        
        cd ../..
    else
        echo "⚠️  git-lfs not found. Moondream model will be downloaded at runtime."
        echo "   To pre-download, install git-lfs and run:"
        echo "   git lfs install"
        echo "   cd models/moondream && git clone https://huggingface.co/vikhyatk/moondream2"
    fi
else
    echo "✅ Moondream model already exists"
fi

echo ""
echo "📊 Model Status:"
echo "   YOLO11n: $([ -f "models/yolo/yolo11n.pt" ] && echo "✅ Available" || echo "❌ Missing")"
echo "   Moondream2: $([ -f "models/moondream/moondream2/model.safetensors" ] && echo "✅ Available" || echo "⚠️  Will download at runtime")"

echo ""
echo "🎯 Models setup complete! You can now run:"
echo "   docker compose build --no-cache"
echo "   docker compose up"

