# Vision Pipeline - Current State & Architecture

## Overview
This document describes the current state of the Vision Pipeline project as of September 2024, including our development workflow, deployment strategy, and current challenges.

## Architecture Summary

### Development Environment
- **Mac Development Host**: Used for development, testing, and repository management
- **NVIDIA Jetson Orin Nano**: Target deployment platform for production vision pipeline

### Current Pipeline Flow
```
Mock Camera (30fps) → Redis → YOLO Detection → Moondream VLM → Fusion → WebSocket API → Frontend
```

## Current Status

### ✅ Working Components
1. **Mock Camera Service**: Generates synthetic 1920x1080 frames at 30fps with animated shapes
2. **Redis Message Bus**: Successfully handles frame and message routing between services  
3. **API Service**: WebSocket connections working, frontend displays frames
4. **Frontend**: Displays camera feed, YOLO detections, VLM chat interface
5. **Fusion Service**: Combines YOLO + VLM results (after recent fixes)

### ❌ Current Issues
1. **YOLO Service**: `ultralytics` package not installed → "Model not loaded, skipping frame processing"
2. **Moondream VLM**: `transformers` package not installed → "Model not loaded, skipping frame processing"  
3. **Real Camera**: CSI camera not yet working (using mock camera for now)

## Model Management Strategy

### Problem
- Models are large (GB+ sizes) and excluded from Git via `.gitignore`
- Models exist on Mac dev host but not synced to Jetson
- Need models available on Jetson for containerized ML services

### Solution: Host-Downloaded + Bind-Mounted Models
1. **On Jetson**: Run `./scripts/setup-models.sh` to download models to host filesystem
   - Downloads `yolo11n.pt` to `./models/yolo/`
   - Downloads Moondream2 to `./models/moondream/moondream2/`

2. **Docker Compose**: Bind-mount host model directories into containers
   ```yaml
   yolo:
     volumes:
       - ./models/yolo:/app/models/yolo
   moondream:  
     volumes:
       - ./models/moondream:/app/models/moondream
   ```

3. **Container Access**: ML services access models via `/app/models/` paths inside containers

## Development Workflow

### Mac Development Host
1. Code development and testing
2. Repository management (Git commits/pushes)
3. Frontend development and debugging
4. Architecture design and documentation

### Jetson Deployment
1. `git pull` to sync latest code changes
2. `./scripts/setup-models.sh` to download/update models (if needed)
3. `docker compose build --no-cache` to rebuild containers with latest code
4. `docker compose up` to run complete pipeline

## Next Steps (Priority Order)

### 1. Fix ML Package Installation
**Problem**: `ultralytics` and `transformers` packages commented out in Dockerfiles due to "puccinialin" build errors
**Solution**: Re-enable package installation with better error handling and fallbacks

### 2. Verify Model Access
**Problem**: Containers may not be accessing bind-mounted models correctly
**Solution**: Check model paths, permissions, and bind-mount configuration

### 3. Test Complete ML Pipeline
**Goal**: Mock Camera → Real YOLO Detection → Real Moondream VLM → Fusion → Frontend
**Success Criteria**: 
- YOLO detects synthetic shapes in mock camera frames
- Moondream VLM responds to chat queries about frames
- Frontend displays real bounding boxes and confidence scores

### 4. Transition to Real Camera
**Goal**: Replace mock camera with Jetson CSI camera
**Challenges**: GStreamer, driver compatibility, hardware setup

## Technical Notes

### Container Strategy
- **Base Images**: `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` (NVIDIA L4T optimized for Jetson)
- **GPU Access**: All ML containers use `runtime: nvidia` for CUDA acceleration
- **Networking**: Custom Docker network for inter-service communication
- **Persistence**: Redis data and model files persist via volumes/bind-mounts

### Key Files
- `docker-compose.yml`: Service orchestration and bind-mount configuration
- `scripts/setup-models.sh`: Model download script for Jetson host
- `containers/*/Dockerfile.jetson`: Container definitions for Jetson deployment
- `services/native/*`: Containerized ML services (YOLO, Moondream, Camera)
- `services/api/*`: WebSocket API service for frontend communication

### Performance Characteristics
- **Mock Camera**: 30fps frame generation (optimized NumPy operations)
- **Message Bus**: ~40ms frame processing latency through Redis
- **Frontend**: Real-time frame display and WebSocket communication
- **ML Services**: Currently in mock mode, real performance TBD

## Known Limitations
1. **ML Packages**: Currently disabled due to build issues
2. **Real Camera**: CSI camera integration pending
3. **GPU Utilization**: ML services not yet using CUDA acceleration
4. **Error Recovery**: Limited resilience to service failures
5. **Monitoring**: No comprehensive health checks or metrics collection

---

*Last Updated: September 8, 2024*
*Next Review: After ML package installation and real model testing*
