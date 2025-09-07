# Vision Pipeline Jetson Deployment Guide

## Overview

This guide covers deploying the complete Vision Pipeline using a **fully containerized architecture** specifically optimized for NVIDIA Jetson devices. All services run in Docker containers with full CUDA GPU acceleration support.

## Architecture

### Containerized Design

**Containerized Services (Infrastructure):**
- `redis`: Message bus for inter-service communication
- `api`: FastAPI service with WebSocket support  
- `frontend`: React-based web interface
- `fusion`: Combines outputs from detection services

**Containerized CV Services (NVIDIA CUDA GPU Access):**
- `camera_service.py`: Camera capture with CSI/USB support
- `yolo_service.py`: YOLO11 object detection using NVIDIA CUDA GPU
- `moondream_service.py`: Moondream2 VLM using NVIDIA CUDA GPU

### Why Fully Containerized?

- NVIDIA provides Docker runtime support for GPU access on Jetson devices
- All services can run in containers while maintaining full CUDA acceleration
- Simplified deployment with no native service management required
- Better resource isolation and easier scaling

## Prerequisites

- **JetPack** 4.6+ (Nano) or 5.0+ (Xavier/Orin)
- **Docker** with NVIDIA runtime support
- **NVIDIA Jetson** device (Nano, Xavier NX/AGX, or Orin)
- **At least 4GB RAM** (8GB+ recommended)
- **Compatible camera** (CSI camera like IMX219-83 Stereo Binocular or USB webcam)

## Complete Setup Guide

### Step 1: Clone and Navigate to Project

```bash
git clone <repository-url>
cd Vision-Pipeline-Jetson
```

### Step 2: Configure NVIDIA Docker Runtime

```bash
# Install NVIDIA Docker runtime (if not already installed)
sudo apt update
sudo apt install -y nvidia-docker2

# Configure Docker daemon
sudo nano /etc/docker/daemon.json
# Add: {"default-runtime": "nvidia", "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}}}

# Restart Docker
sudo systemctl restart docker

# Verify NVIDIA runtime
docker run --runtime=nvidia --rm nvcr.io/nvidia/l4t-base:r35.2.1 nvidia-smi
```

### Step 3: Optimize Jetson Performance

```bash
# Set maximum performance mode
sudo jetson_clocks
sudo nvpmodel -m 0  # Maximum performance mode (if available)

# For Jetson Nano - increase swap space
if [[ $(cat /proc/device-tree/model) == *"Nano"* ]]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi
```

**System optimizations applied:**
- Maximum CPU/GPU clock speeds
- Increased swap space (Nano only)
- NVIDIA Docker runtime configured
- Performance mode enabled

### Step 4: Setup Model Files

The project includes pre-downloaded model files. Verify they exist:

```bash
# Check YOLO model
ls -la models/yolo/yolo11n.pt

# Check Moondream model files
ls -la models/moondream/moondream2/
```

**Important**: If you move or clone this project to a different directory, you may need to fix HuggingFace cache symlinks:

```bash
# Remove broken symlinks (if they exist)
rm ~/.cache/huggingface/modules/transformers_modules/moondream2/*.py 2>/dev/null || true

# Copy actual files to cache
cp models/moondream/moondream2/*.py ~/.cache/huggingface/modules/transformers_modules/moondream2/ 2>/dev/null || true
```

### Step 5: Configure Environment (Optional)

```bash
# Copy example environment file
cp env.example .env

# Edit configuration if needed
nano .env
```

**Key configuration options:**
```bash
# Camera Configuration
CAMERA_INDEX=0
CAMERA_WIDTH=1920
CAMERA_HEIGHT=1080
CAMERA_FPS=6                    # Optimized for processing pipeline

# YOLO Configuration  
YOLO_MODEL=yolo11n.pt
YOLO_DEVICE=mps                 # Apple Silicon GPU
YOLO_CONFIDENCE=0.5
YOLO_FRAME_STRIDE=2             # Process every 2nd frame

# Moondream Configuration
VLM_FRAME_STRIDE=10             # Process every 10th frame
VLM_MAX_CONTEXT_LENGTH=100

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Step 6: Build and Start All Services

```bash
# Build and start all containerized services
docker-compose up --build -d

# Monitor startup logs
docker-compose logs -f
```

### Step 7: Start the Complete System

```bash
# Start all services (hybrid architecture)
./scripts/start-all.sh
```

**What this script does:**
1. Checks prerequisites (Docker, virtual environment, base image)
2. Stops any existing services
3. Starts Docker services (Redis, API, Frontend, Fusion)
4. Starts native services (Camera, YOLO, Moondream)
5. Verifies all services are running

## Accessing the Application

Once startup is complete, access:

- **Frontend UI**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Service Management

### Check Status
```bash
./scripts/status.sh
```

### View Logs
```bash
# View all logs in real-time
./scripts/logs-unified.sh

# View specific service logs
tail -f logs/camera_native.log
tail -f logs/yolo_native.log
tail -f logs/moondream_native.log

# View Docker service logs
docker-compose logs -f api
docker-compose logs -f frontend
```

### Stop All Services
```bash
./scripts/stop-all.sh
```

### Restart Individual Services

**Native services:**
```bash
# Stop service
pkill -f camera_native    # or yolo_native, moondream_native

# Start service
source models/yolo11_env/bin/activate
python3 services/native/camera_native.py > logs/camera_native.log 2>&1 &
```

**Docker services:**
```bash
docker-compose restart api
docker-compose restart frontend
```

## Performance Characteristics

**Apple Silicon Performance:**
- Camera: ~6 FPS capture rate
- YOLO: ~50 FPS capability (limited by camera)
- Moondream: 1-3 seconds per VLM query
- End-to-end latency: <100ms (detection), 1-3s (VLM)

**Scaling by Mac Model:**
- M1: Good performance for development
- M1 Pro/Max: Excellent for production use
- M2/M3/M4: Optimal performance across all models

## Troubleshooting

### Common Issues

#### 1. Camera Service Won't Start
**Error**: `ModuleNotFoundError: No module named 'cv2'`

**Solution**:
```bash
# Recreate virtual environment
rm -rf models/yolo11_env
python3 -m venv models/yolo11_env
source models/yolo11_env/bin/activate
pip install -r containers/requirements-base.txt
```

#### 2. Moondream Model Won't Load
**Error**: `No such file or directory: hf_moondream.py`

**Solution**:
```bash
# Fix HuggingFace cache symlinks
mkdir -p ~/.cache/huggingface/modules/transformers_modules/moondream2
rm ~/.cache/huggingface/modules/transformers_modules/moondream2/*.py 2>/dev/null || true
cp models/moondream/moondream2/*.py ~/.cache/huggingface/modules/transformers_modules/moondream2/

# Restart Moondream service
pkill -f moondream_native
source models/yolo11_env/bin/activate
python3 services/native/moondream_native.py > logs/moondream_native.log 2>&1 &
```

#### 3. Docker Services Won't Start
**Error**: Docker daemon not running

**Solution**:
```bash
# Start Docker Desktop application
open -a Docker

# Wait for Docker to start, then retry
./scripts/start-all.sh
```

#### 4. Port Already in Use
**Error**: Port 3000 or 8000 already in use

**Solution**:
```bash
# Find and kill processes using the ports
lsof -ti:3000 | xargs kill -9
lsof -ti:8000 | xargs kill -9

# Or modify docker-compose.yml to use different ports
```

#### 5. Virtual Environment Path Issues
**Error**: Virtual environment created in wrong location

**Solution**:
```bash
# Remove and recreate in correct location
rm -rf models/yolo11_env
cd /path/to/Vision-Pipeline-Mac  # Ensure you're in project root
python3 -m venv models/yolo11_env
source models/yolo11_env/bin/activate
pip install -r containers/requirements-base.txt
```

### Performance Tuning

#### Optimize for Lower Latency
```bash
# Edit .env or export environment variables
export CAMERA_FPS=10
export YOLO_FRAME_STRIDE=1
export VLM_FRAME_STRIDE=5
```

#### Optimize for Lower Resource Usage
```bash
export CAMERA_FPS=6
export YOLO_FRAME_STRIDE=3
export VLM_FRAME_STRIDE=15
export CAMERA_WIDTH=1280
export CAMERA_HEIGHT=720
```

### Log Analysis

**Structured JSON logs** are used throughout. Use `jq` for analysis:

```bash
# Parse camera service logs
tail -f logs/camera_native.log | jq '.'

# Filter for errors
tail -f logs/moondream_native.log | jq 'select(.level=="ERROR")'

# Monitor performance metrics
tail -f logs/yolo_native.log | jq 'select(.message | contains("Performance"))'
```

## Development Workflow

### Making Changes

1. **Frontend changes**: Automatically reloaded in development
2. **API changes**: Restart API container: `docker-compose restart api`
3. **Native service changes**: Stop and restart specific service
4. **Configuration changes**: Restart affected services

### Testing Changes

```bash
# Quick health check
curl http://localhost:8000/health

# Test WebSocket connection
# Use browser developer tools or WebSocket test tools

# Verify camera feed
# Check frontend at http://localhost:3000
```

## Updating the System

### Update Dependencies
```bash
# Update Python packages
source models/yolo11_env/bin/activate
pip install --upgrade -r containers/requirements-base.txt

# Rebuild Docker images
./scripts/build-base.sh
docker-compose build
```

### Update Models
```bash
# Update YOLO model
cd models/yolo
# Download new model file

# Update Moondream model
# Download new model files to models/moondream/moondream2/
```

## Data and Logs

### Log Locations
- **Native services**: `logs/*.log` (JSON formatted)
- **Docker services**: `docker-compose logs <service>`

### Model Cache
- **YOLO models**: `models/yolo/`
- **Moondream models**: `models/moondream/`
- **HuggingFace cache**: `~/.cache/huggingface/`

### Backup Important Data
```bash
# Backup configuration
tar -czf vision-pipeline-backup.tar.gz .env docker-compose.yml models/ logs/

# Backup virtual environment (optional)
tar -czf venv-backup.tar.gz models/yolo11_env/
```

## Production Considerations

### Security
- Change default ports in `docker-compose.yml`
- Configure proper CORS settings in API
- Use environment secrets for sensitive data
- Enable TLS/SSL for external access

### Monitoring
- Set up log aggregation (ELK stack, Grafana)
- Monitor resource usage with `docker stats`
- Set up health check endpoints
- Configure alerts for service failures

### Scaling
- Multiple camera inputs require separate native services
- API and Frontend can be scaled horizontally
- Redis can be clustered for high availability

## Support and Maintenance

### Regular Maintenance
```bash
# Clean up Docker resources
docker system prune

# Rotate logs (implement log rotation)
# Monitor disk space in logs/ directory

# Update system regularly
brew upgrade
pip list --outdated
```

### Getting Help
1. Check service logs using commands above
2. Verify system status with `./scripts/status.sh`
3. Test API endpoints at http://localhost:8000/docs
4. Monitor resource usage with `docker stats` and Activity Monitor

## Quick Reference

### Essential Commands
```bash
# Start everything
./scripts/start-all.sh

# Stop everything  
./scripts/stop-all.sh

# Check status
./scripts/status.sh

# View all logs
./scripts/logs-unified.sh

# Restart native service
pkill -f <service>_native
source models/yolo11_env/bin/activate
python3 services/native/<service>_native.py > logs/<service>_native.log 2>&1 &
```

### File Structure
```
Vision-Pipeline-Mac/
├── scripts/                    # Management scripts
│   ├── start-all.sh           # Complete startup
│   ├── stop-all.sh            # Complete shutdown
│   └── status.sh              # Status check
├── services/native/           # Native Apple Silicon services
│   ├── camera_native.py       # Camera capture
│   ├── yolo_native.py         # Object detection
│   └── moondream_native.py    # VLM processing
├── models/                    # Model files and virtual env
│   ├── yolo11_env/           # Python virtual environment
│   ├── yolo/                 # YOLO models
│   └── moondream/            # Moondream models
├── logs/                     # Service logs
├── docker-compose.yml        # Container orchestration
└── containers/               # Docker configurations
    └── requirements-base.txt # Python dependencies
```

This hybrid architecture provides optimal performance for Apple Silicon Macs while maintaining clean separation between infrastructure and ML processing services.