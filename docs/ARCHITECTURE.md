# Moondream Vision Pipeline Architecture

## Hybrid Container/Native Architecture

This project uses a **hybrid architecture** that combines Docker containers with native services to optimize for different hardware platforms.

### Apple Silicon (macOS) Development Setup

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
- Docker on macOS runs in a Linux VM without direct GPU access
- Native services can leverage Metal Performance Shaders (MPS)

### Other Architectures (Jetson Nano, Thor, etc.)

For platforms with direct Docker GPU access, all services can run in containers:

1. **Uncomment services in `docker-compose.yml`:**
   - `camera` service
   - `yolo` service  
   - `moondream` service

2. **Uncomment dependencies in `fusion` service:**
   - `yolo: condition: service_started`
   - `moondream: condition: service_started`

3. **Use standard Docker Compose:**
   ```bash
   DOCKER_BUILDKIT=0 docker-compose up --build
   ```

### Service Communication

All services communicate via **Redis pub/sub channels**:
- `frames`: Camera publishes raw frame data
- `detections.yolo`: YOLO publishes bounding boxes
- `detections.vlm`: Moondream publishes descriptions  
- `chat.requests`: Frontend sends VLM chat requests
- `chat.responses`: Moondream sends chat responses

### Development Commands

**Apple Silicon:**
```bash
./scripts/start-all.sh    # Start hybrid architecture
./scripts/stop-all.sh     # Stop all services
./scripts/status.sh       # Check service status
```

**Other Architectures:**
```bash
# Edit docker-compose.yml first (uncomment services)
DOCKER_BUILDKIT=0 docker-compose up --build
```

### Performance Characteristics

**Apple Silicon Native:**
- Camera: ~6 FPS
- YOLO: ~6 FPS (every frame)
- Moondream: ~0.6 FPS (every 10th frame)

**Container Performance:**
- Varies by hardware platform
- GPU acceleration available on Jetson/Thor
- CPU-only on most cloud platforms

## File Structure

```
├── docker-compose.yml          # Container definitions (hybrid config)
├── scripts/
│   ├── start-all.sh            # Apple Silicon startup
│   ├── stop-all.sh             # Stop all services
│   └── status.sh               # Status check
├── services/
│   ├── api/                    # FastAPI service (container)
│   ├── cv_services/            # Fusion service (container)
│   ├── native/                 # Native services (Apple Silicon)
│   └── shared/                 # Common models/utils
└── containers/                 # Dockerfiles for all services
```

This architecture provides optimal performance on Apple Silicon while maintaining compatibility with other platforms.
