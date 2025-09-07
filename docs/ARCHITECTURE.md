# Vision Pipeline Mac Architecture

## Hybrid Architecture for Apple Silicon

This project uses a **hybrid architecture** specifically optimized for Apple Silicon Macs, combining Docker containers with native services to leverage GPU acceleration.

### Architecture Overview

**Containerized Services:**
- `redis`: Message bus for inter-service communication
- `api`: FastAPI service with WebSocket support  
- `frontend`: React-based web interface
- `fusion`: Combines outputs from detection services

**Native Services (for GPU access):**
- `camera`: Camera capture and frame publishing
- `yolo`: YOLO11 object detection using Apple Silicon GPU
- `moondream`: Moondream2 VLM using Apple Silicon GPU

**Why Hybrid?**
- Apple Silicon GPU access requires native execution
- Docker Desktop on macOS runs in a Linux VM without Metal Performance Shaders access
- Native services leverage Apple's Metal Performance Shaders for maximum performance
- Containerized infrastructure services provide easy management and portability

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

**Apple Silicon Performance:**
- Camera: ~6 FPS capture rate
- YOLO: ~50 FPS capability (limited by camera)
- Moondream: 1-3 seconds per VLM query
- End-to-end latency: <100ms (detection), 1-3s (VLM)

**Scaling by Mac Model:**
- M1: Good performance for development
- M1 Pro/Max: Excellent for production use
- M2/M3/M4: Optimal performance across all models

## File Structure

```
Vision-Pipeline-Mac/
├── docker-compose.yml          # Hybrid architecture configuration
├── scripts/
│   ├── start-all.sh            # Complete startup script
│   ├── quick-start.sh          # Fast startup script
│   ├── stop-all.sh             # Stop all services
│   └── status.sh               # Status check
├── services/
│   ├── api/                    # FastAPI service (container)
│   ├── native/                 # Native services (Apple Silicon)
│   │   ├── camera_native.py    # Camera capture
│   │   ├── yolo_native.py      # YOLO detection
│   │   └── moondream_native.py # VLM processing
│   ├── message_bus/            # Redis pub/sub implementation
│   └── shared/                 # Common models/utils
├── frontend/                   # React UI (container)
├── containers/                 # Dockerfiles for containerized services
└── models/                     # Model storage
    ├── moondream/              # Moondream2 VLM
    ├── yolo/                   # YOLO11 models
    └── yolo11_env/             # Python environment
```

This architecture provides optimal performance specifically for Apple Silicon Macs by leveraging native GPU acceleration while maintaining clean separation of concerns.
