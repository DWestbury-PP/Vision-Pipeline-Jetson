# Moondream Vision Pipeline Deployment Guide

## Overview

This guide covers deploying the complete Moondream Vision Pipeline using Docker containers. The system is designed to run entirely in containers without requiring any packages to be installed on the host system.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 8GB RAM (16GB recommended for multiple Moondream instances)
- Compatible camera device (Mac Studio Display camera supported out of the box)

## Quick Start

1. **Clone and Navigate to the Project**
   ```bash
   cd /path/to/moondream-vision-pipeline
   ```

2. **Configure Environment**
   ```bash
   # Copy and customize environment file
   cp env.example .env
   # Edit .env file with your specific settings
   ```

3. **Deploy All Services**
   ```bash
   ./scripts/deploy.sh
   ```

4. **Access the Application**
   - Frontend UI: http://localhost:3000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## Architecture

The system consists of the following containerized services:

- **Redis**: Message bus for high-performance pub/sub
- **Camera Service**: Captures frames and publishes to message bus
- **YOLO Service**: Fast object detection (YOLO11)
- **Moondream Service**: VLM processing with parallel instances
- **Fusion Service**: Combines YOLO and VLM outputs
- **API Service**: REST API and WebSocket handlers
- **Frontend**: React UI with real-time visualization

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Camera Configuration
CAMERA_TYPE=mac_studio          # mac_studio, jetson, usb
CAMERA_WIDTH=1920
CAMERA_HEIGHT=1080
CAMERA_FPS=30

# YOLO Configuration
YOLO_MODEL=yolo11n.pt          # Model size: n, s, m, l, x
YOLO_CONFIDENCE=0.5
YOLO_DEVICE=mps                # mps (Apple Silicon), cuda, cpu

# Moondream Configuration
MOONDREAM_MODEL=vikhyatk/moondream2
MOONDREAM_DEVICE=mps           # mps (Apple Silicon), cuda, cpu
MOONDREAM_INSTANCES=2          # Number of parallel instances

# Pipeline Configuration
VLM_FRAME_STRIDE=10           # Process every Nth frame with VLM
YOLO_FRAME_STRIDE=1           # Process every Nth frame with YOLO
```

### Hardware-Specific Settings

#### Apple Silicon (M1/M2/M3/M4)
```bash
YOLO_DEVICE=mps
MOONDREAM_DEVICE=mps
```

#### NVIDIA GPU
```bash
YOLO_DEVICE=cuda
MOONDREAM_DEVICE=cuda
```

#### CPU Only
```bash
YOLO_DEVICE=cpu
MOONDREAM_DEVICE=cpu
```

## Deployment Commands

### Full Deployment
```bash
./scripts/deploy.sh
```

### Manual Deployment
```bash
# Start Redis first
docker-compose up -d redis

# Build and start all services
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f
```

### Individual Service Management
```bash
# Start specific service
docker-compose up -d camera

# View service logs
docker-compose logs -f moondream

# Restart service
docker-compose restart yolo

# Scale Moondream instances
docker-compose up -d --scale moondream=3
```

## Monitoring and Debugging

### Health Checks
```bash
# Check all services
docker-compose ps

# API health check
curl http://localhost:8000/health

# System status
curl http://localhost:8000/status
```

### Log Analysis
```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f camera
docker-compose logs -f yolo
docker-compose logs -f moondream

# View structured logs with jq
docker-compose logs camera | jq '.'
```

### Performance Monitoring
```bash
# Container resource usage
docker stats

# API performance metrics
curl http://localhost:8000/metrics
```

## Troubleshooting

### Common Issues

#### Camera Access Issues
```bash
# Ensure camera device access
ls -la /dev/video*

# Check camera service logs
docker-compose logs camera
```

#### GPU Access Issues
```bash
# For NVIDIA GPU
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# For Apple Silicon
# MPS support is automatic in PyTorch containers
```

#### Memory Issues
```bash
# Check container memory usage
docker stats

# Reduce Moondream instances
# Edit .env: MOONDREAM_INSTANCES=1
docker-compose up -d --scale moondream=1
```

#### Network Issues
```bash
# Check port availability
netstat -ln | grep 3000
netstat -ln | grep 8000

# Restart with different ports
# Edit docker-compose.yml port mappings
```

### Performance Tuning

#### Optimize for Latency
- Reduce `VLM_FRAME_STRIDE` (process more frames)
- Use faster YOLO model (yolo11n.pt)
- Increase `MOONDREAM_INSTANCES`

#### Optimize for Throughput
- Increase `VLM_FRAME_STRIDE` (process fewer frames)
- Use larger YOLO model (yolo11l.pt)
- Optimize camera resolution

#### Memory Optimization
- Reduce `MOONDREAM_INSTANCES`
- Lower camera resolution
- Use smaller models

## Updating the System

### Update All Services
```bash
# Pull latest images
docker-compose pull

# Rebuild services
docker-compose build

# Restart with new images
docker-compose up -d
```

### Update Individual Components
```bash
# Update frontend only
docker-compose build frontend
docker-compose up -d frontend

# Update models
docker-compose exec yolo python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

## Data Persistence

### Model Cache
Models are cached in Docker volumes:
- `model_cache`: YOLO models
- `huggingface_cache`: Moondream models

### Logs
Structured logs are output to stdout and can be collected using:
```bash
# Save logs to file
docker-compose logs > system.log

# Use external log aggregation
# Configure logging driver in docker-compose.yml
```

## Security Considerations

### Production Deployment
- Change default ports
- Configure proper CORS settings
- Use environment secrets
- Enable TLS/SSL
- Restrict network access

### Example Production Configuration
```yaml
# docker-compose.prod.yml
services:
  api:
    environment:
      - API_HOST=0.0.0.0
      - CORS_ORIGINS=https://yourdomain.com
    networks:
      - internal
  
  frontend:
    environment:
      - REACT_APP_API_BASE_URL=https://api.yourdomain.com
```

## Backup and Recovery

### Backup Configuration
```bash
# Backup environment and docker-compose files
tar -czf moondream-config-backup.tar.gz .env docker-compose.yml

# Backup model cache
docker run --rm -v model_cache:/data -v $(pwd):/backup alpine tar czf /backup/models-backup.tar.gz -C /data .
```

### Recovery
```bash
# Restore configuration
tar -xzf moondream-config-backup.tar.gz

# Restore models
docker run --rm -v model_cache:/data -v $(pwd):/backup alpine tar xzf /backup/models-backup.tar.gz -C /data
```

## Support

For issues and questions:
1. Check the logs using the commands above
2. Review the API documentation at http://localhost:8000/docs
3. Monitor system status at http://localhost:8000/status
4. Check container health with `docker-compose ps`
