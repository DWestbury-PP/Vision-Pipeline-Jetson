# Vision Pipeline Jetson Architecture

## Fully Containerized Architecture for NVIDIA Jetson

This project uses a **fully containerized architecture** specifically optimized for NVIDIA Jetson devices, with all services running in Docker containers that have access to CUDA GPU acceleration.

### Architecture Overview

**Containerized Services:**
- `redis`: Message bus for inter-service communication
- `api`: FastAPI service with WebSocket support  
- `frontend`: React-based web interface
- `fusion`: Combines outputs from detection services

**Containerized CV Services (with GPU access):**
- `camera`: Camera capture and frame publishing with CSI/USB support
- `yolo`: YOLO11 object detection using NVIDIA CUDA GPU
- `moondream`: Moondream2 VLM using NVIDIA CUDA GPU

**Why Fully Containerized?**
- NVIDIA provides Docker runtime support for GPU access on Jetson
- All services can run in containers while maintaining CUDA acceleration
- Simplified deployment and management with complete containerization
- Better resource isolation and scalability

### Service Communication

All services communicate via **Redis pub/sub channels**:
- `frames`: Camera publishes raw frame data
- `detections.yolo`: YOLO publishes bounding boxes
- `detections.vlm`: Moondream publishes descriptions  
- `chat.requests`: Frontend sends VLM chat requests
- `chat.responses`: Moondream sends chat responses

### Development Commands

```bash
./scripts/start-all.sh    # Start hybrid architecture
./scripts/quick-start.sh  # Quick start (minimal output)
./scripts/stop-all.sh     # Stop all services
./scripts/status.sh       # Check service status
./scripts/logs-unified.sh # View all logs
```

### Performance Characteristics

**NVIDIA Jetson Performance:**
- Camera: ~10-30 FPS capture rate (device dependent)
- YOLO: ~30-100 FPS capability (device dependent)
- Moondream: 1-2 seconds per VLM query
- End-to-end latency: <50ms (detection), 1-2s (VLM)

**Scaling by Jetson Model:**
- Nano: Good performance for development and lightweight applications
- Xavier NX: Excellent for production edge deployment
- Xavier AGX/Orin: Optimal performance for demanding applications

## File Structure

```
Vision-Pipeline-Jetson/
├── docker-compose.yml          # Containerized architecture configuration
├── scripts/
│   ├── start-all.sh            # Complete startup script
│   ├── quick-start.sh          # Fast startup script
│   ├── stop-all.sh             # Stop all services
│   └── status.sh               # Status check
├── services/
│   ├── api/                    # FastAPI service (container)
│   ├── cv_services/            # Computer vision services (containers with CUDA)
│   │   ├── camera_service.py   # Camera capture with CSI support
│   │   ├── yolo_service.py     # YOLO detection with CUDA
│   │   └── moondream_service.py # VLM processing with CUDA
│   ├── message_bus/            # Redis pub/sub implementation
│   └── shared/                 # Common models/utils
├── frontend/                   # React UI (container)
├── containers/                 # Dockerfiles for containerized services
└── models/                     # Model storage
    ├── moondream/              # Moondream2 VLM
    ├── yolo/                   # YOLO11 models
    └── jetson_models/          # Optimized models for Jetson
```

This architecture provides optimal performance specifically for NVIDIA Jetson devices by leveraging containerized CUDA GPU acceleration while maintaining clean separation of concerns.
