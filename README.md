# Moondream Vision Pipeline 🌙

A modular, high-performance computer vision pipeline featuring real-time object detection (YOLO11) and Vision Language Model (VLM) capabilities powered by Moondream2. Built with a hybrid architecture optimized for Apple Silicon Macs.

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/docker-required-blue.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-lightgrey.svg)

## 🎯 Features

- **Live Camera Feed**: Real-time video capture from your camera (Apple Studio Display, webcam, etc.)
- **Object Detection**: YOLO11 with bounding boxes and confidence scores
- **Vision Language Model**: Moondream2 for intelligent scene understanding and chat
- **Interactive Chat**: Ask questions about what the camera sees
- **Modern UI**: Clean, responsive interface with shadcn components
- **Modular Architecture**: Microservices connected via Redis message bus
- **Hybrid Deployment**: Containerized infrastructure with native ML services for GPU access

## 🏗️ Architecture

### Hybrid Architecture (macOS)
Due to Docker Desktop limitations on macOS (no GPU access), we use a hybrid approach:

**Containerized Services** (Docker):
- Redis (Message Bus)
- API (FastAPI + WebSocket)
- Frontend (React + Vite)
- Fusion (Optional aggregation service)

**Native Services** (Python):
- Camera Capture (OpenCV)
- YOLO11 (Object Detection)
- Moondream2 (VLM)

All services communicate via Redis Pub/Sub for seamless integration.

## 📋 Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.9+**
- **Docker Desktop** for Mac
- **Git** and **Git LFS**
- ~8GB free disk space for models

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/moondream-vision.git
cd moondream-vision
```

### 2. Download Models
```bash
# Create model directories
mkdir -p models/moondream models/yolo

# Download Moondream2 (one-time setup, ~2GB)
cd models/moondream
git lfs install
git clone https://huggingface.co/vikhyatk/moondream2
cd ../..

# YOLO11 will auto-download on first run
```

### 3. Setup Python Environment
```bash
# Create virtual environment
python3 -m venv models/yolo11_env
source models/yolo11_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install opencv-python-headless pillow numpy
pip install ultralytics torch torchvision
pip install redis pydantic pydantic-settings
pip install transformers accelerate
```

### 4. Build Base Docker Image
```bash
# First-time setup (builds optimized base image)
./scripts/build-base.sh
```

### 5. Start the Pipeline
```bash
# Quick start (minimal output)
./scripts/quick-start.sh

# OR full start with detailed output
./scripts/start-all.sh
```

### 6. Access the Application
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## 💻 Usage

### Starting Services
```bash
# Quick start (recommended for development)
./scripts/quick-start.sh

# Full start with health checks
./scripts/start-all.sh

# Check status
./scripts/status.sh

# View unified logs
./scripts/logs-unified.sh
```

### Stopping Services
```bash
./scripts/stop-all.sh
```

### UI Controls
- **Mirror Mode**: Horizontally flip the camera feed
- **YOLO Detection**: Toggle object detection overlays
- **VLM Processing**: Enable/disable Moondream processing
- **Chat Interface**: Ask questions about what the camera sees

### Example Chat Queries
- "What do you see?"
- "Describe the person in the image"
- "What objects are on the desk?"
- "What is the person doing?"
- "Describe the scene in detail"

## 🔧 Configuration

### Environment Variables

**Camera Service**:
- `CAMERA_FPS`: Frame rate (default: 6)
- `CAMERA_WIDTH`: Resolution width (default: 1920)
- `CAMERA_HEIGHT`: Resolution height (default: 1080)

**YOLO Service**:
- `YOLO_CONFIDENCE`: Detection threshold (default: 0.5)
- `YOLO_DEVICE`: Compute device (default: mps for Apple Silicon)

**Moondream Service**:
- `VLM_FRAME_STRIDE`: Process every Nth frame (default: 10)

## 📁 Project Structure

```
moondream-vision/
├── services/
│   ├── api/              # FastAPI REST & WebSocket server
│   ├── frontend/         # React UI with Vite
│   ├── native/           # Native Python services
│   │   ├── camera_native.py
│   │   ├── yolo_native.py
│   │   └── moondream_native.py
│   ├── message_bus/      # Redis Pub/Sub implementation
│   └── shared/           # Shared models and utilities
├── containers/           # Dockerfiles
├── scripts/              # Management scripts
├── models/               # Model storage (gitignored)
│   ├── moondream/        # Moondream2 model files
│   └── yolo11_env/       # Python virtual environment
├── logs/                 # Service logs (gitignored)
└── docker-compose.yml    # Container orchestration
```

## 🐛 Troubleshooting

### Camera Not Working
- Ensure camera permissions are granted in System Settings
- Check `logs/camera_native.log` for errors
- Try different `CAMERA_INDEX` values (0, 1, 2)

### Model Loading Issues
- Verify models are downloaded to `models/` directory
- Check available disk space
- Ensure virtual environment is activated

### WebSocket Connection Failed
- Clear browser cache and refresh
- Check if API is running: `docker ps | grep api`
- Verify Redis is healthy: `docker exec moondream-redis redis-cli ping`

### Performance Issues
- Reduce `CAMERA_FPS` to 3-4
- Increase `YOLO_FRAME_STRIDE` to skip more frames
- Increase `VLM_FRAME_STRIDE` for less frequent VLM processing

## 🔍 Monitoring

```bash
# View all logs
./scripts/logs-unified.sh

# Check specific service
tail -f logs/camera_native.log
tail -f logs/yolo_native.log
tail -f logs/moondream_native.log

# Docker logs
docker-compose logs api --follow
docker-compose logs frontend --follow

# Redis monitoring
docker exec moondream-redis redis-cli MONITOR
```

## 🎯 Performance

Typical performance on Apple Silicon Mac:
- Camera: 6 FPS capture rate
- YOLO11: 15-20ms per detection
- Moondream2: 2-3 seconds per VLM query
- WebSocket latency: <10ms
- End-to-end pipeline: <100ms for detection, 2-3s for VLM

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Moondream2](https://github.com/vikhyat/moondream) by Vikhyat Korrapati
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [shadcn/ui](https://ui.shadcn.com/) for UI components
- OpenCV and PyTorch communities

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review logs in `logs/` directory

---

**Status**: ✅ Fully Operational | Last Updated: September 2025