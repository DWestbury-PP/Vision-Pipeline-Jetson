# Moondream Vision Pipeline

A high-performance, modular computer vision pipeline designed for **robotics applications**. Features real-time object detection (YOLO11) and Vision Language Model capabilities (Moondream2) with a sophisticated messaging architecture optimized for both reactive and deliberative AI systems.

![Status](https://img.shields.io/badge/status-production-success.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/docker-required-blue.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20|%20Jetson%20|%20Linux-lightgrey.svg)

## Vision Pipeline in Action

![Moondream Vision Pipeline UI](docs/Screen-Capture.png)

The interface demonstrates the **dual-channel architecture**: fast YOLO detection (green bounding boxes) providing immediate object recognition, while the VLM chat delivers rich, contextual understanding of the scene.

## Key Features

### **Dual-Speed Processing Architecture**
- **Fast Channel**: YOLO11 object detection (~6 FPS) for reactive responses
- **Deep Channel**: Moondream2 VLM (~0.6 FPS) for scene understanding and chat
- **Optimized for Robotics**: Multi-tier processing (reactive → deliberative → cognitive)

### **Real-Time Vision Pipeline**
- **Live Camera Feed**: Apple Studio Display, USB cameras, or Jetson CSI cameras
- **Interactive Bounding Boxes**: Real-time object detection with confidence scores
- **Natural Language Interface**: Chat with the AI about what it sees
- **Modern Dark UI**: Professional interface optimized for computer vision work

### **Scalable Messaging Architecture**
- **Redis Pub/Sub**: High-performance message bus with frame compression
- **Asynchronous Processing**: Non-blocking pipeline for maximum throughput
- **Modular Design**: Easy integration of additional vision models
- **Performance Monitoring**: Built-in metrics and logging

## Platform Support

### **Apple Silicon (Primary Development Platform)**
**Current State**: Fully operational with hybrid architecture
- **GPU Acceleration**: Native services leverage Apple Silicon GPU (Metal Performance Shaders)
- **Containerized Infrastructure**: Redis, API, Frontend run in Docker
- **Native ML Services**: Camera, YOLO, Moondream run natively for GPU access

### **NVIDIA Jetson Nano/Thor (Robotics Target)**
**Status**: Architecture ready, requires configuration
- **Full Containerization**: All services can run in Docker with GPU support
- **NVIDIA Runtime**: Configured for CUDA acceleration
- **CSI Camera Support**: Direct hardware camera access
- **Edge Optimization**: Designed for resource-constrained robotics applications

### **Linux/Cloud Platforms**
**Status**: Supported with CPU inference
- **Docker Compose**: Standard containerized deployment
- **CPU Inference**: Fallback for platforms without GPU acceleration
- **Scalable**: Multi-instance deployment for high-throughput applications

## Architecture Overview

### **Hybrid Architecture (Apple Silicon)**
```
┌─────────────────┐    ┌──────────────────┐
│   Containerized │    │      Native      │
│                 │    │                  │
│ • Redis         │◄──►│ • Camera         │
│ • API           │    │ • YOLO11         │
│ • Frontend      │    │ • Moondream2     │
│ • Fusion        │    │                  │
└─────────────────┘    └──────────────────┘
```

### **Full Container Architecture (Jetson/Linux)**
```
┌─────────────────────────────────────────┐
│              Docker Compose             │
│                                         │
│ • Redis     • API      • Frontend       │
│ • Camera    • YOLO11   • Moondream2     │
│ • Fusion                                │
└─────────────────────────────────────────┘
```

### **Message Flow Architecture**
```
Camera → Redis → YOLO11 ───┐
              └→ Moondream ┴→ Fusion → API → WebSocket → Frontend
                     ↑                              ↓
                     └──── Chat Requests ←──────────┘
```

## Prerequisites

### **Apple Silicon Mac**
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- Docker Desktop for Mac
- ~8GB free disk space for models

### **NVIDIA Jetson Nano/Thor**
- JetPack 4.6+ (Ubuntu 18.04/20.04)
- Docker with NVIDIA runtime configured
- Python 3.9+
- ~8GB free storage

### **Linux/Cloud**
- Ubuntu 20.04+ or equivalent
- Docker and Docker Compose
- Python 3.9+
- Optional: NVIDIA GPU with CUDA support

## Quick Start

### **Apple Silicon Mac (Hybrid Mode)**

```bash
# 1. Clone repository
git clone https://github.com/DWestbury-PP/Moondream.git
cd Moondream

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

### **NVIDIA Jetson (Full Container Mode)**

```bash
# 1. Configure Docker for GPU support
sudo nano /etc/docker/daemon.json
# Add:
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}

sudo systemctl restart docker

# 2. Clone and setup
git clone https://github.com/DWestbury-PP/Moondream.git
cd Moondream

# 3. Uncomment Jetson services in docker-compose.yml
# (Uncomment camera, yolo, moondream services and their dependencies)

# 4. Start with Docker Compose
DOCKER_BUILDKIT=0 docker-compose up --build

# 5. Access the application
curl http://localhost:3000
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

### **Apple Silicon Mac Studio (M2 Ultra)**
- **Camera**: 6 FPS capture
- **YOLO11**: 15-20ms per detection (~50 FPS capable)
- **Moondream2**: 2-3 seconds per VLM query
- **End-to-end latency**: <100ms (detection), 2-3s (VLM)
- **Memory usage**: ~4GB (models loaded)

### **NVIDIA Jetson Nano (4GB)**
- **Camera**: 5 FPS capture  
- **YOLO11**: 50-80ms per detection (~12 FPS)
- **Moondream2**: 8-12 seconds per VLM query
- **Memory usage**: ~3.5GB (optimized models)

### **NVIDIA Jetson Thor (Estimated)**
- **Camera**: 10+ FPS capture
- **YOLO11**: 10-15ms per detection (~60 FPS)
- **Moondream2**: 1-2 seconds per VLM query
- **Multi-model support**: 3-5 concurrent vision models

## Robotics Integration

### **Adding New Vision Models**

```python
# 1. Create new service
class NewVisionService:
    def __init__(self):
        self.frame_stride = 5  # Process every 5th frame
        
    async def process_frame(self, frame, metadata):
        # Your model inference here
        results = await self.model.predict(frame)
        
        # Publish to message bus
        await self.publisher.publish_message(
            "detection.new_model", 
            results
        )

# 2. Add to docker-compose.yml
new_vision:
  build: ./containers/new_vision
  environment:
    - REDIS_HOST=redis
    - MODEL_STRIDE=5
```

### **Recommended Processing Rates**

| **Vision Task** | **Rate (FPS)** | **Robotics Use Case** |
|----------------|----------------|----------------------|
| Obstacle Detection | 10-30 | Navigation safety |
| Object Recognition | 5-10 | Task planning |
| Scene Understanding | 1-2 | Context awareness |
| Facial Recognition | 2-5 | Human interaction |
| Text/OCR | 0.5-1 | Information extraction |
| Depth Estimation | 5-15 | Spatial reasoning |

## Project Structure

```
Moondream/
├── services/
│   ├── api/                 # FastAPI + WebSocket server
│   ├── native/              # Apple Silicon native services
│   │   ├── camera_native.py
│   │   ├── yolo_native.py
│   │   └── moondream_native.py
│   ├── message_bus/         # Redis pub/sub implementation
│   └── shared/              # Common models and utilities
├── frontend/                # React + TypeScript UI
├── containers/              # Dockerfiles for all services
├── scripts/                 # Management and deployment scripts
├── docs/                    # Architecture and API documentation
├── models/                  # Model storage (gitignored)
└── docker-compose.yml       # Multi-platform orchestration
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

### **Development (Apple Silicon)**
```bash
./scripts/start-all.sh
```

### **Production (Jetson/Linux)**
```bash
# Edit docker-compose.yml (uncomment services)
DOCKER_BUILDKIT=0 docker-compose up -d --scale yolo=2 --scale moondream=1
```

### **Cloud/Distributed**
```bash
# Multi-node deployment with load balancing
docker swarm init
docker stack deploy -c docker-compose.prod.yml moondream-stack
```

## Contributing

We welcome contributions! This project is designed to become the foundation for advanced robotics vision systems.

**Priority Areas:**
- Additional vision models (depth estimation, segmentation)
- Jetson optimization and testing
- Multi-camera support
- Real-time performance improvements
- Robotics-specific integrations

**Development Process:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Test on your target platform
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

- **Issues**: [GitHub Issues](https://github.com/DWestbury-PP/Moondream/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DWestbury-PP/Moondream/discussions)
- **Documentation**: [Project Wiki](https://github.com/DWestbury-PP/Moondream/wiki)

---
