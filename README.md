# Moondream Vision Pipeline

A modular, high-performance computer vision pipeline combining YOLO11 object detection with Moondream VLM for comprehensive visual understanding.

## Architecture Overview

This solution provides a scalable pipeline that:
- Captures camera input with modular camera abstraction
- Uses a high-performance message bus for low-latency image distribution
- Supports multiple CV/VLM subscribers (YOLO11, Moondream instances)
- Provides a polished web UI with real-time visualization
- Enables chat-based VLM interactions
- Supports horizontal scaling with parallel VLM instances

## Components

- **Camera Service**: Modular camera capture with architecture-specific adapters
- **Message Bus**: High-performance pub/sub for image distribution  
- **YOLO Service**: Fast object detection for continuous feedback
- **Moondream Service**: VLM instances for rich semantic understanding
- **Fusion Service**: Combines outputs from multiple CV/VLM sources
- **Web UI**: React frontend with real-time visualization and chat
- **API Gateway**: WebSocket and REST API coordination

## Quick Start

```bash
# Build and start all services
docker-compose up --build

# Access the UI
open http://localhost:3000
```

## Development

See `docs/DEVELOPMENT.md` for detailed development instructions.

## Deployment Targets

- Mac Mini M4 (initial development)
- Jetson Orin/Thor (future deployment)
- Other ARM64/x86_64 architectures
