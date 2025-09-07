# Vision Pipeline Mac

A high-performance, modular computer vision pipeline optimized for **Apple Silicon Macs**. Features real-time object detection (YOLO11) and Vision Language Model capabilities (Moondream2) with a hybrid architecture that leverages native GPU acceleration for maximum performance.

![Status](https://img.shields.io/badge/status-production-success.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/docker-required-blue.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-success.svg)

## Vision Pipeline in Action

![Vision Pipeline Mac UI](docs/Screen-Capture.png)

The interface demonstrates the **hybrid architecture**: fast YOLO detection (green bounding boxes) providing immediate object recognition, while the VLM chat delivers rich, contextual understanding of the scene. Optimized for Apple Silicon's Metal Performance Shaders.

## Key Features

### **Hybrid Architecture for Apple Silicon**
- **Native GPU Acceleration**: YOLO11 and Moondream2 leverage Metal Performance Shaders
- **Containerized Infrastructure**: Redis, API, and Frontend run in Docker
- **Optimized Performance**: ~6 FPS YOLO detection, 2-3s VLM responses

### **Real-Time Vision Pipeline**
- **Live Camera Feed**: Apple Studio Display, USB webcams, built-in cameras
- **Interactive Bounding Boxes**: Real-time object detection with confidence scores
- **Natural Language Interface**: Chat with the AI about what it sees
- **Modern Dark UI**: Professional interface optimized for Mac users

### **Scalable Messaging Architecture**
- **Redis Pub/Sub**: High-performance message bus with frame compression
- **Asynchronous Processing**: Non-blocking pipeline for maximum throughput
- **Modular Design**: Easy integration of additional vision models
- **Performance Monitoring**: Built-in metrics and logging

## System Requirements

### **Apple Silicon Mac (M1/M2/M3/M4)**
- **macOS**: Monterey (12.0) or later
- **Memory**: 16GB RAM recommended (8GB minimum)
- **Storage**: 8GB free space for models
- **Camera**: Built-in, Apple Studio Display, or USB webcam
- **GPU**: Automatic Metal Performance Shaders acceleration

## Architecture Overview

### **Hybrid Architecture**
```
┌─────────────────┐    ┌──────────────────┐
│   Docker        │    │   Native macOS   │
│                 │    │                  │
│ • Redis         │◄──►│ • Camera         │
│ • API           │    │ • YOLO11 (MPS)   │
│ • Frontend      │    │ • Moondream2     │
│ • Fusion        │    │                  │
└─────────────────┘    └──────────────────┘
```

### **Message Flow**
```
Camera → Redis → YOLO11 ───┐
              └→ Moondream ┴→ API → WebSocket → Frontend
                     ↑                         ↓
                     └──── Chat Requests ←─────┘
```

**Why Hybrid?** Docker Desktop on macOS doesn't support GPU access, so ML models run natively to leverage Apple Silicon's Metal Performance Shaders while infrastructure services run in containers for easy management.

## Prerequisites

- **macOS**: Monterey (12.0) or later
- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4)
- **Python**: 3.9 or later
- **Docker Desktop**: Latest version for Mac
- **Storage**: ~8GB free disk space for models
- **Memory**: 8GB RAM minimum, 16GB recommended

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/DWestbury-PP/Vision-Pipeline-Mac.git
cd Vision-Pipeline-Mac

# 2. Download models
mkdir -p models/moondream models/yolo
cd models/moondream
git lfs install
git clone https://huggingface.co/vikhyatk/moondream2
cd ../..

# 3. Setup Python environment
python3 -m venv models/yolo11_env
source models/yolo11_env/bin/activate
pip install --upgrade pip
pip install opencv-python-headless pillow numpy
pip install ultralytics torch torchvision
pip install redis pydantic pydantic-settings
pip install transformers accelerate

# 4. Start the pipeline
./scripts/start-all.sh

# 5. Access the application
open http://localhost:3000
```

## Usage

### **Control Interface**

**YOLO Detection Panel:**
- **Toggle**: Enable/disable object detection
- **Bounding Boxes**: Show/hide detection overlays  
- **Confidence Scores**: Display detection confidence
- **Real-time Status**: Detection count and processing status

**VLM Chat Interface:**
- **Toggle**: Enable/disable VLM processing
- **Natural Language Chat**: Ask questions about the scene
- **Processing Indicators**: Shows when VLM is analyzing
- **iPhone-style Scrolling**: Conversation history with smooth scrolling

**Camera Controls:**
- **Mirror Mode**: Horizontal flip for selfie-style viewing
- **Connection Status**: Real-time connection indicators
- **Frame Information**: Resolution, FPS, and frame counters

### **Example Chat Interactions**

```
User: "What do you see in this image?"
VLM:  "I can see a person sitting at a desk with a computer setup. 
       There's a keyboard, mouse, and what appears to be a coffee 
       cup on the desk. The person is wearing glasses and appears 
       to be working."

User: "What objects are on the desk?"
VLM:  "On the desk I can identify several objects: a black keyboard,
       a computer mouse, a white coffee mug, some papers or documents,
       and what looks like a smartphone. There's also a computer 
       monitor visible in the background."
```

## Configuration & Optimization

### **Performance Tuning**

**Frame Processing Rates:**
```bash
# Camera capture rate
export CAMERA_FPS=6          # Balanced performance

# YOLO processing (every Nth frame)
export YOLO_FRAME_STRIDE=1   # Process every frame

# VLM processing (every Nth frame)  
export VLM_FRAME_STRIDE=10   # Process every 10th frame
```

**Resource Optimization:**
```bash
# For lower-end hardware
export CAMERA_FPS=3
export YOLO_FRAME_STRIDE=2
export VLM_FRAME_STRIDE=20

# For high-performance setups
export CAMERA_FPS=10
export YOLO_FRAME_STRIDE=1
export VLM_FRAME_STRIDE=5
```

### **Robotics-Specific Settings**

**Multi-Tier Processing:**
```bash
# Reactive tier (< 100ms)
export OBSTACLE_DETECTION_FPS=30
export EMERGENCY_STOP_FPS=60

# Deliberative tier (100ms - 1s)
export OBJECT_CLASSIFICATION_FPS=10
export PATH_PLANNING_FPS=5

# Cognitive tier (1s+)
export SCENE_UNDERSTANDING_FPS=1
export TASK_PLANNING_FPS=0.5
```

## Performance Benchmarks

### **Apple Silicon Performance**

| **Mac Model** | **YOLO11** | **Moondream2** | **Memory Usage** |
|---------------|------------|----------------|------------------|
| **M1 Mac** | 25-30ms | 3-4 seconds | ~3.5GB |
| **M1 Pro/Max** | 20-25ms | 2.5-3 seconds | ~4GB |
| **M2** | 18-22ms | 2-2.5 seconds | ~4GB |
| **M2 Pro/Max** | 15-20ms | 1.5-2 seconds | ~4.5GB |
| **M3/M4** | 12-18ms | 1-1.5 seconds | ~4.5GB |

### **Real-World Performance**
- **Camera Capture**: 6 FPS (configurable)
- **YOLO Detection**: ~50 FPS capable (limited by camera)
- **End-to-end Latency**: <100ms (detection), 1-3s (VLM)
- **Concurrent Processing**: Both models run simultaneously without interference

## Extending the Pipeline

### **Adding New Vision Models**

The modular architecture makes it easy to add new vision models:

```python
# 1. Create new native service (services/native/new_model_native.py)
class NewModelService:
    def __init__(self):
        self.device = "mps"  # Apple Silicon GPU
        self.frame_stride = 5  # Process every 5th frame
        
    async def process_frame(self, frame, metadata):
        # Your model inference here
        results = await self.model.predict(frame)
        
        # Publish to Redis
        await self.publisher.publish_message(
            "detection.new_model", 
            results
        )

# 2. Add startup script
./scripts/start-native-newmodel.sh
```

### **Recommended Processing Rates for Mac**

| **Vision Task** | **Apple Silicon Rate** | **Use Case** |
|----------------|------------------------|--------------|
| Object Detection | 30-60 FPS | Real-time tracking |
| Scene Understanding | 1-3 FPS | Context analysis |
| Facial Recognition | 10-20 FPS | Human interaction |
| Text/OCR | 2-5 FPS | Document processing |
| Depth Estimation | 10-30 FPS | Spatial analysis |

## Project Structure

```
Vision-Pipeline-Mac/
├── services/
│   ├── api/                 # FastAPI + WebSocket server
│   ├── native/              # Apple Silicon native services
│   │   ├── camera_native.py
│   │   ├── yolo_native.py
│   │   └── moondream_native.py
│   ├── message_bus/         # Redis pub/sub implementation
│   └── shared/              # Common models and utilities
├── frontend/                # React + TypeScript UI
├── containers/              # Dockerfiles for containerized services
├── scripts/                 # Management and deployment scripts
├── docs/                    # Architecture and API documentation
├── models/                  # Model storage (gitignored)
│   ├── moondream/           # Moondream2 VLM
│   ├── yolo/                # YOLO11 models
│   └── yolo11_env/          # Python virtual environment
└── docker-compose.yml       # Hybrid architecture orchestration
```

## Monitoring & Debugging

### **Real-time Monitoring**
```bash
# Unified log viewer
./scripts/logs-unified.sh

# Service-specific logs
tail -f logs/camera_native.log
tail -f logs/yolo_native.log
tail -f logs/moondream_native.log

# Redis message monitoring
docker exec moondream-redis redis-cli MONITOR

# Performance metrics
./scripts/status.sh
```

### **Common Issues & Solutions**

**Camera Not Detected:**
```bash
# Check camera permissions (macOS)
# System Settings → Privacy & Security → Camera

# Test camera access
python3 -c "import cv2; print(cv2.VideoCapture(0).read())"

# Try different camera indices
export CAMERA_INDEX=1  # or 2, 3, etc.
```

**GPU Not Accessible:**
```bash
# Apple Silicon - verify MPS
python3 -c "import torch; print(torch.backends.mps.is_available())"

# NVIDIA - verify CUDA
docker run --runtime=nvidia --rm nvidia/cuda:11.0-base nvidia-smi
```

**Model Loading Failures:**
```bash
# Check model files
ls -la models/moondream/moondream2/
ls -la models/yolo/

# Verify disk space
df -h

# Re-download models
rm -rf models/moondream/moondream2
cd models/moondream && git clone https://huggingface.co/vikhyatk/moondream2
```

## Deployment Options

### **Development Mode**
```bash
# Quick start (minimal output)
./scripts/quick-start.sh

# Full start with detailed output
./scripts/start-all.sh
```

### **Production Mode**
```bash
# Optimized for performance
export CAMERA_FPS=10
export YOLO_FRAME_STRIDE=1
export VLM_FRAME_STRIDE=5
./scripts/start-all.sh
```

### **Resource-Constrained Mode**
```bash
# For older Macs or limited memory
export CAMERA_FPS=3
export YOLO_FRAME_STRIDE=2
export VLM_FRAME_STRIDE=15
./scripts/start-all.sh
```

## Contributing

We welcome contributions! This project demonstrates high-performance computer vision on Apple Silicon.

**Priority Areas:**
- Additional vision models (depth estimation, segmentation, pose detection)
- Multi-camera support for Apple Studio Display + webcam
- Performance optimizations for different Mac models
- UI/UX improvements for the React frontend
- Integration with Core ML models

**Development Process:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Test on your Apple Silicon Mac
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Documentation

- **[Architecture Overview](docs/ARCHITECTURE.md)**: Hybrid vs container deployment
- **[Messaging Backend](docs/MESSAGING_BACKEND.md)**: Redis pub/sub and performance optimization
- **[API Documentation](http://localhost:8000/docs)**: Interactive API explorer (when running)

## Acknowledgments

- **[Moondream2](https://github.com/vikhyat/moondream)** by Vikhyat Korrapati - Exceptional VLM model
- **[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)** - State-of-the-art object detection
- **[shadcn/ui](https://ui.shadcn.com/)** - Beautiful, accessible UI components
- **OpenCV, PyTorch, Redis communities** - Foundation technologies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/DWestbury-PP/Vision-Pipeline-Mac/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DWestbury-PP/Vision-Pipeline-Mac/discussions)
- **Documentation**: [Project Wiki](https://github.com/DWestbury-PP/Vision-Pipeline-Mac/wiki)

---
