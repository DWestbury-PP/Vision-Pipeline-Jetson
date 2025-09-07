# Vision Pipeline Jetson - Repository Blueprint

This document outlines the structure and modifications needed to create a **Vision Pipeline Jetson** repository from the current **Vision Pipeline Mac** codebase.

## Repository Overview

**Target Repository**: `Vision-Pipeline-Jetson`
**Purpose**: Fully containerized vision pipeline optimized for NVIDIA Jetson devices
**Architecture**: All services run in Docker containers with GPU acceleration

## Key Differences from Mac Version

### 1. **Full Containerization**
- **Mac Version**: Hybrid (containers + native services)
- **Jetson Version**: All services in Docker containers

### 2. **GPU Access**
- **Mac Version**: Native services for Metal Performance Shaders
- **Jetson Version**: NVIDIA Docker runtime for CUDA acceleration

### 3. **Camera Support**
- **Mac Version**: USB webcams, Apple Studio Display
- **Jetson Version**: CSI cameras, USB cameras

## Required Changes

### docker-compose.yml Modifications

```yaml
# Uncomment and modify these services:
camera:
  build:
    context: .
    dockerfile: containers/camera/Dockerfile.jetson
  container_name: vision-pipeline-camera
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
    - CAMERA_TYPE=jetson_csi
    - CAMERA_INDEX=0
  privileged: true
  volumes:
    - /dev:/dev

yolo:
  build:
    context: .
    dockerfile: containers/yolo/Dockerfile.jetson
  container_name: vision-pipeline-yolo
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
    - YOLO_DEVICE=cuda
  volumes:
    - ./models/yolo:/app/models/yolo

moondream:
  build:
    context: .
    dockerfile: containers/moondream/Dockerfile.jetson
  container_name: vision-pipeline-vlm
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
    - MOONDREAM_DEVICE=cuda
  volumes:
    - moondream_cache:/home/appuser/.cache/huggingface
```

### New Dockerfiles Required

#### containers/camera/Dockerfile.jetson
```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Install camera dependencies for Jetson
RUN apt-get update && apt-get install -y \
    python3-opencv \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.jetson.txt .
RUN pip install -r requirements.jetson.txt

# Copy service code
COPY services/camera/ /app/
WORKDIR /app

CMD ["python3", "camera_jetson.py"]
```

#### containers/yolo/Dockerfile.jetson
```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Install YOLO dependencies
RUN pip install ultralytics

# Copy service code
COPY services/yolo/ /app/
COPY models/yolo/ /app/models/yolo/
WORKDIR /app

CMD ["python3", "yolo_jetson.py"]
```

#### containers/moondream/Dockerfile.jetson
```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Install transformers and dependencies
RUN pip install transformers accelerate

# Copy service code
COPY services/moondream/ /app/
WORKDIR /app

CMD ["python3", "moondream_jetson.py"]
```

### New Service Files Required

#### services/camera/camera_jetson.py
- CSI camera support using `cv2.VideoCapture` with GStreamer pipeline
- V4L2 device detection and configuration
- Jetson-specific camera optimizations

#### services/yolo/yolo_jetson.py
- CUDA device initialization
- TensorRT optimization for Jetson
- Memory management for edge devices

#### services/moondream/moondream_jetson.py
- CUDA device configuration
- Model quantization for edge deployment
- Memory-efficient inference

### README Updates for Jetson

```markdown
# Vision Pipeline Jetson

A high-performance, fully containerized computer vision pipeline optimized for **NVIDIA Jetson devices**. Features real-time object detection (YOLO11) and Vision Language Model capabilities (Moondream2) with CUDA acceleration.

## System Requirements

### **NVIDIA Jetson Nano (4GB)**
- **JetPack**: 4.6+ (Ubuntu 18.04/20.04)
- **Memory**: 4GB RAM (optimized models)
- **Storage**: 32GB+ microSD card (64GB recommended)
- **Camera**: CSI camera or USB webcam

### **NVIDIA Jetson Xavier NX/AGX**
- **JetPack**: 5.0+ (Ubuntu 20.04)
- **Memory**: 8GB+ RAM
- **Storage**: 32GB+ eMMC or NVMe SSD
- **Camera**: CSI camera or USB webcam

### **NVIDIA Jetson Orin (Thor)**
- **JetPack**: 5.1+ (Ubuntu 20.04/22.04)
- **Memory**: 32GB+ RAM
- **Storage**: NVMe SSD recommended
- **Camera**: Multiple CSI/USB cameras supported

## Quick Start

```bash
# 1. Configure Docker for GPU support
sudo nano /etc/docker/daemon.json
# Add NVIDIA runtime configuration

# 2. Clone repository
git clone https://github.com/DWestbury-PP/Vision-Pipeline-Jetson.git
cd Vision-Pipeline-Jetson

# 3. Download models (lightweight versions for edge)
./scripts/download-models-jetson.sh

# 4. Start the pipeline
DOCKER_BUILDKIT=0 docker-compose up --build

# 5. Access the application
curl http://localhost:3000
```
```

### Performance Optimizations for Jetson

#### Model Optimizations
- **YOLO**: Use YOLOv8n or YOLOv11n (nano) models
- **Moondream**: Quantized INT8 models for faster inference
- **TensorRT**: Convert models to TensorRT for maximum performance

#### Memory Management
- **Swap Configuration**: Increase swap space on Nano
- **Model Caching**: Efficient model loading and caching
- **Garbage Collection**: Aggressive memory cleanup

#### Power Management
- **Jetson Clocks**: Set maximum performance mode
- **Thermal Management**: Monitor and throttle if necessary
- **Power Modes**: Configure appropriate power mode for use case

## Scripts to Create

### scripts/setup-jetson.sh
```bash
#!/bin/bash
# Configure Jetson for optimal performance
sudo jetson_clocks
sudo nvpmodel -m 0  # Maximum performance mode

# Configure Docker
sudo systemctl restart docker

# Set up swap (for Nano)
if [[ $(cat /proc/device-tree/model) == *"Nano"* ]]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
fi
```

### scripts/download-models-jetson.sh
```bash
#!/bin/bash
# Download optimized models for Jetson

mkdir -p models/moondream models/yolo

# Download lightweight YOLO model
cd models/yolo
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt

# Download quantized Moondream model (if available)
cd ../moondream
# Use HuggingFace Hub with quantized models
```

## Repository Structure

```
Vision-Pipeline-Jetson/
├── docker-compose.yml          # Full container configuration
├── scripts/
│   ├── setup-jetson.sh         # Jetson optimization
│   ├── download-models-jetson.sh
│   └── performance-test.sh
├── services/
│   ├── camera/
│   │   └── camera_jetson.py    # CSI camera support
│   ├── yolo/
│   │   └── yolo_jetson.py      # CUDA-optimized YOLO
│   ├── moondream/
│   │   └── moondream_jetson.py # CUDA-optimized VLM
│   └── shared/                 # Common utilities
├── containers/
│   ├── camera/
│   │   └── Dockerfile.jetson   # Jetson camera container
│   ├── yolo/
│   │   └── Dockerfile.jetson   # Jetson YOLO container
│   └── moondream/
│       └── Dockerfile.jetson   # Jetson VLM container
└── docs/
    ├── JETSON_SETUP.md         # Detailed setup guide
    ├── PERFORMANCE_TUNING.md   # Optimization guide
    └── TROUBLESHOOTING.md      # Common issues
```

## Migration Steps

1. **Fork/Clone** the Vision-Pipeline-Mac repository
2. **Rename** repository to Vision-Pipeline-Jetson
3. **Update** docker-compose.yml (uncomment services, add NVIDIA runtime)
4. **Create** Jetson-specific Dockerfiles
5. **Implement** Jetson-optimized service files
6. **Add** setup and optimization scripts
7. **Update** README and documentation
8. **Test** on target Jetson hardware
9. **Optimize** for performance and memory usage

## Key Benefits of Separate Repository

1. **Clean Deployment**: No Mac-specific code or configurations
2. **Focused Documentation**: Jetson-specific setup and troubleshooting
3. **Optimized Dependencies**: Edge-specific model versions and libraries
4. **Hardware-Specific Features**: CSI cameras, power management, thermal controls
5. **Community Focus**: Jetson developers can contribute without Mac complexity

This blueprint provides a complete roadmap for creating a production-ready Jetson version of the vision pipeline.
