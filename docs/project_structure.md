# Project Structure

```
moondream/
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
│
├── docs/
│   ├── PLANNING.md
│   ├── DEVELOPMENT.md
│   ├── DEPLOYMENT.md
│   └── API.md
│
├── services/
│   ├── camera/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract camera interface
│   │   ├── mac_studio.py        # Mac Studio Display camera
│   │   ├── jetson.py           # Jetson camera implementations
│   │   └── config.py
│   │
│   ├── message_bus/
│   │   ├── __init__.py
│   │   ├── redis_bus.py        # Redis pub/sub implementation
│   │   ├── zmq_bus.py          # ZeroMQ alternative
│   │   └── base.py
│   │
│   ├── cv_services/
│   │   ├── __init__.py
│   │   ├── yolo_service.py     # YOLO11 detection service
│   │   ├── moondream_service.py # Moondream VLM service
│   │   └── fusion_service.py   # Output fusion service
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── websocket_handler.py
│   │   ├── rest_api.py
│   │   └── models.py           # Pydantic models
│   │
│   └── shared/
│       ├── __init__.py
│       ├── logging_config.py   # Structured logging
│       ├── config.py          # Configuration management
│       ├── models.py          # Shared data models
│       └── utils.py
│
├── frontend/
│   ├── package.json
│   ├── tailwind.config.js
│   ├── components.json         # shadcn config
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/            # shadcn components
│   │   │   ├── CameraView.tsx
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── ControlPanel.tsx
│   │   │   └── BoundingBoxOverlay.tsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts
│   │   │   └── useCamera.ts
│   │   ├── types/
│   │   │   └── index.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   └── public/
│
├── containers/
│   ├── camera/Dockerfile
│   ├── yolo/Dockerfile
│   ├── moondream/Dockerfile
│   ├── api/Dockerfile
│   └── frontend/Dockerfile
│
└── scripts/
    ├── setup.sh
    ├── build.sh
    └── deploy.sh
```

## Service Architecture

### Camera Service
- Abstract base class for camera implementations
- Mac Studio Display camera implementation
- Jetson camera implementations
- Configurable frame rates and resolutions

### Message Bus
- Redis pub/sub for high-performance message distribution
- ZeroMQ alternative for different deployment scenarios
- Frame metadata and image reference publishing

### CV Services
- **YOLO Service**: Fast object detection (every frame or subset)
- **Moondream Service**: VLM processing (keyframes, chat queries)
- **Fusion Service**: Combines and reconciles outputs

### API Layer
- WebSocket for real-time communication
- REST API for configuration and control
- Structured data models with Pydantic

### Frontend
- React with TypeScript
- shadcn/ui components for polished UI
- Real-time WebSocket integration
- Camera view with bounding box overlays
- Chat interface for VLM interaction

### Containerization
- Individual containers for each service
- Docker Compose orchestration
- Environment-specific configurations
