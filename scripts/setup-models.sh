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
    
    # Try multiple YOLO download sources
    echo "🔍 Trying official Ultralytics release..."
    if command -v wget >/dev/null 2>&1; then
        # Try the official Ultralytics GitHub releases
        wget -O yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt || \
        wget -O yolo11n.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt || \
        wget -O yolo11n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    elif command -v curl >/dev/null 2>&1; then
        # Try with curl
        curl -L -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt || \
        curl -L -o yolo11n.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt || \
        curl -L -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    else
        echo "❌ Neither wget nor curl found. Please install one of them or download manually:"
        echo "   Try one of these URLs and save as models/yolo/yolo11n.pt:"
        echo "   - https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
        echo "   - https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
        echo "   - Or use: pip install ultralytics && python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\""
    fi
    
    # Check if download was successful
    if [ -f "yolo11n.pt" ]; then
        echo "✅ YOLO model downloaded successfully"
    else
        echo "⚠️  YOLO download failed. Trying Python approach..."
        echo "💡 Alternative: Use Python to download:"
        echo "   python3 -c \"from ultralytics import YOLO; model = YOLO('yolo11n.pt'); print('Model downloaded')\""
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
        
        # Initialize git-lfs if not already done
        git lfs install 2>/dev/null || true
        
        # Clone the model repository
        if [ ! -d "moondream2" ]; then
            echo "🔄 Cloning Moondream2 repository..."
            git clone https://huggingface.co/vikhyatk/moondream2
            if [ $? -eq 0 ]; then
                echo "✅ Moondream model downloaded successfully"
            else
                echo "⚠️  Git clone failed. Model will be downloaded at runtime."
            fi
        else
            echo "📁 Moondream repository already exists, pulling latest..."
            cd moondream2
            git pull || echo "⚠️  Git pull failed, using existing model"
            cd ..
        fi
        
        cd ../..
    else
        echo "⚠️  git-lfs not found. Installing git-lfs..."
        # Try to install git-lfs
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y git-lfs
        elif command -v brew >/dev/null 2>&1; then
            brew install git-lfs
        else
            echo "📋 Please install git-lfs manually:"
            echo "   Ubuntu/Debian: sudo apt-get install git-lfs"
            echo "   macOS: brew install git-lfs"
            echo "   Then run: git lfs install"
            echo "   cd models/moondream && git clone https://huggingface.co/vikhyatk/moondream2"
        fi
    fi
else
    echo "✅ Moondream model already exists"
fi

echo ""
echo "📊 Model Status:"
echo "   YOLO11n: $([ -f "models/yolo/yolo11n.pt" ] && echo "✅ Available ($(du -h models/yolo/yolo11n.pt | cut -f1))" || echo "❌ Missing")"
echo "   Moondream2: $([ -f "models/moondream/moondream2/model.safetensors" ] && echo "✅ Available" || echo "⚠️  Will download at runtime")"

echo ""
echo "🎯 Models setup complete! You can now run:"
echo "   docker compose build --no-cache"
echo "   docker compose up"

echo ""
echo "💡 If models are missing, you can also:"
echo "   - Use Python: python3 -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\""
echo "   - Download manually from HuggingFace: https://huggingface.co/vikhyatk/moondream2"

