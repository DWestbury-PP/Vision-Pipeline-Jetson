# Moondream Vision Pipeline

A modular, high-performance computer vision pipeline combining YOLO11 object detection with Moondream VLM for comprehensive visual understanding. Built with containerized microservices for scalability and rapid development.

## 🚀 Architecture Overview

This solution provides a scalable, containerized pipeline that:
- **Modular Camera Capture**: Architecture-specific camera adapters (Mac Studio, Jetson, USB)
- **High-Performance Message Bus**: Redis pub/sub for low-latency image distribution
- **Multiple CV/VLM Subscribers**: YOLO11 for fast detection, Moondream for rich understanding
- **Professional Web UI**: React frontend with shadcn components, real-time visualization
- **Chat-Based VLM Interactions**: Natural language queries to the vision system
- **Horizontal Scaling**: Multiple parallel VLM instances for improved throughput
- **Fast Development**: Optimized Docker setup with 99.7% faster builds (25s vs 15+ minutes)

## 🏗️ Components

| Service | Purpose | Technology |
|---------|---------|------------|
| **Camera Service** | Modular camera capture with architecture adapters | OpenCV, asyncio |
| **Message Bus** | High-performance pub/sub for image distribution | Redis |
| **YOLO Service** | Fast object detection for continuous feedback | YOLO11, PyTorch |
| **Moondream Service** | VLM instances for rich semantic understanding | Moondream VLM |
| **Fusion Service** | Combines outputs from multiple CV/VLM sources | Python |
| **API Service** | WebSocket and REST API coordination | FastAPI |
| **Frontend** | React UI with real-time visualization and chat | React, Vite, shadcn/ui |

## ⚡ Quick Start

### Prerequisites
- Docker Desktop
- 8GB+ RAM recommended
- Camera access (for full functionality)

### Fast Development Setup

```bash
# 1. Build optimized base image (one-time, ~1 minute)
./scripts/build-base.sh

# 2. Start development environment (25 seconds)
./scripts/fast-dev.sh

# 3. Access the applications
open http://localhost:3000  # Frontend UI
open http://localhost:8000/docs  # API Documentation
```

### Individual Service Testing

```bash
# Start specific services for testing
./scripts/fast-dev.sh redis api frontend --build -d

# View logs
docker-compose logs api --follow
```

## 🛠️ Development Workflow

### Fast Iteration Cycle
The optimized build system enables rapid development:

1. **Base Image**: Contains all system dependencies and Python packages
2. **Service Images**: Only copy code changes (rebuilds in seconds)
3. **Hot Reloading**: Frontend supports live reloading
4. **Structured Logging**: JSON logs across all services

### Configuration
- Environment variables in `env.example`
- Pydantic-based configuration with validation
- Container-specific environment overrides

### Architecture Details
See `docs/project_structure.md` for detailed component breakdown.

## 🐳 Container Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Service   │    │  Camera Service │
│   (React/Vite)  │    │   (FastAPI)     │    │   (OpenCV)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Redis Message  │
                    │      Bus        │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  YOLO Service   │    │ Moondream VLM   │    │ Fusion Service  │
│   (Detection)   │    │   (Semantic)    │    │  (Aggregation)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 Features

### UI Features
- **Live Camera Feed**: Real-time video display with optional horizontal mirroring
- **Bounding Box Overlays**: Toggle detection boxes with confidence scores
- **Chat Interface**: Natural language queries to VLM
- **Professional Design**: shadcn/ui components with modern UX
- **Real-time Updates**: WebSocket-based live data streaming

### Performance Features
- **Fast Builds**: 99.7% faster container builds (25s vs 15+ min)
- **Parallel Processing**: Multiple VLM instances for throughput
- **Adaptive Processing**: Frame stride and scene change detection
- **Low Latency**: Optimized message bus and async processing

## 📊 Performance Metrics

- **Build Time**: 25 seconds (vs 15+ minutes previously)
- **Base Image Size**: 4.23GB (includes all dependencies)
- **Service Rebuild**: <5 seconds (code-only changes)
- **Memory Usage**: ~2GB for full pipeline
- **Latency**: <100ms for YOLO detection, <2s for VLM processing

## 🚢 Deployment Targets

- **Development**: Mac Mini M4 with Apple Studio Display camera
- **Production**: Jetson Orin/Thor with architecture-specific camera adapters
- **Cloud**: ARM64/x86_64 container orchestration platforms

## 📝 Documentation

- `docs/DEPLOYMENT.md` - Production deployment guide
- `docs/project_structure.md` - Detailed architecture breakdown
- `docs/PLANNING.md` - Original project planning and requirements

## 🔧 Troubleshooting

### Common Issues

1. **Camera Access in Containers**: Expected limitation - containers can't access host camera by default
2. **Port Conflicts**: Use `lsof -ti:3000 | xargs kill -9` to free ports
3. **Docker Network Issues**: Run `docker network prune -f` to clean up

### Performance Optimization

- Use `./scripts/fast-dev.sh` for development (not docker-compose directly)
- Base image is cached - only rebuild when dependencies change
- Monitor container resources with `docker stats`

## 🤝 Contributing

1. Use fast development workflow for iterations
2. Follow structured logging patterns
3. Update documentation for architectural changes
4. Test with both simulated and real camera inputs

---

**Built for rapid iteration and production deployment** 🚀