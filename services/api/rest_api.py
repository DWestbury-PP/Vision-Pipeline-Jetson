"""REST API for the Moondream Vision Pipeline."""

import asyncio
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import uuid

from .models import (
    SystemStatusResponse, ServiceStatusResponse, ConfigurationUpdateRequest,
    ConfigurationResponse, ChatRequest, CameraControlRequest, CameraControlResponse,
    ServiceControlRequest, ServiceControlResponse, FrameRequest, FrameResponse,
    PerformanceMetrics, HealthCheckResponse, UIConfigRequest, UIConfigResponse,
    ErrorResponse
)
from .websocket_handler import WebSocketHandler
from ..shared.logging_config import setup_logging, log_error_with_context
from ..shared.config import get_config
from ..shared.models import UISettings


class MoondreamAPI:
    """Main API class for the Moondream Vision Pipeline."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("rest_api")
        
        # FastAPI app
        self.app = FastAPI(
            title="Moondream Vision Pipeline API",
            description="REST API for computer vision pipeline with YOLO and Moondream VLM",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # WebSocket handler
        self.websocket_handler = WebSocketHandler()
        
        # Service tracking
        self.start_time = time.time()
        self.ui_settings = UISettings()
        
        # Service references (to be injected)
        self.camera_service = None
        self.yolo_service = None
        self.moondream_service = None
        self.fusion_service = None
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Set up FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Set up API routes."""
        
        @self.app.get("/", response_model=dict)
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "Moondream Vision Pipeline API",
                "version": "1.0.0",
                "status": "running",
                "endpoints": {
                    "docs": "/docs",
                    "health": "/health",
                    "status": "/status",
                    "websocket": "/ws/{client_id}"
                }
            }
        
        @self.app.get("/health", response_model=HealthCheckResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                uptime = time.time() - self.start_time
                
                # Check service health
                services = {}
                if self.camera_service:
                    services["camera"] = "healthy" if hasattr(self.camera_service, 'is_running') and self.camera_service.is_running else "unhealthy"
                if self.yolo_service:
                    services["yolo"] = "healthy" if hasattr(self.yolo_service, 'is_running') and self.yolo_service.is_running else "unhealthy"
                if self.moondream_service:
                    services["moondream"] = "healthy" if hasattr(self.moondream_service, 'is_running') and self.moondream_service.is_running else "unhealthy"
                if self.fusion_service:
                    services["fusion"] = "healthy" if hasattr(self.fusion_service, 'is_running') and self.fusion_service.is_running else "unhealthy"
                
                # Check message bus
                message_bus_health = "healthy"
                try:
                    if hasattr(self.websocket_handler, 'message_bus'):
                        is_healthy = await self.websocket_handler.message_bus.health_check()
                        message_bus_health = "healthy" if is_healthy else "unhealthy"
                except:
                    message_bus_health = "unhealthy"
                
                overall_status = "healthy" if all(status == "healthy" for status in services.values()) and message_bus_health == "healthy" else "degraded"
                
                return HealthCheckResponse(
                    status=overall_status,
                    services=services,
                    message_bus=message_bus_health,
                    uptime_seconds=uptime
                )
                
            except Exception as e:
                log_error_with_context(self.logger, e, operation="health_check")
                raise HTTPException(status_code=500, detail="Health check failed")
        
        @self.app.get("/status", response_model=SystemStatusResponse)
        async def get_system_status():
            """Get comprehensive system status."""
            try:
                uptime = time.time() - self.start_time
                
                # Collect service statuses
                services = []
                
                if self.camera_service:
                    camera_info = await self.camera_service.get_camera_info() if hasattr(self.camera_service, 'get_camera_info') else {}
                    services.append(ServiceStatusResponse(
                        service_name="camera",
                        is_running=getattr(self.camera_service, 'is_running', False),
                        details=camera_info
                    ))
                
                if self.yolo_service:
                    yolo_info = await self.yolo_service.get_service_info() if hasattr(self.yolo_service, 'get_service_info') else {}
                    services.append(ServiceStatusResponse(
                        service_name="yolo",
                        is_running=getattr(self.yolo_service, 'is_running', False),
                        details=yolo_info
                    ))
                
                if self.moondream_service:
                    moondream_info = await self.moondream_service.get_service_info() if hasattr(self.moondream_service, 'get_service_info') else {}
                    services.append(ServiceStatusResponse(
                        service_name="moondream",
                        is_running=getattr(self.moondream_service, 'is_running', False),
                        details=moondream_info
                    ))
                
                if self.fusion_service:
                    fusion_info = await self.fusion_service.get_service_info() if hasattr(self.fusion_service, 'get_service_info') else {}
                    services.append(ServiceStatusResponse(
                        service_name="fusion",
                        is_running=getattr(self.fusion_service, 'is_running', False),
                        details=fusion_info
                    ))
                
                # Create system status (placeholder)
                from ..shared.models import SystemStatus
                system_status = SystemStatus(
                    camera_active=any(s.service_name == "camera" and s.is_running for s in services),
                    yolo_active=any(s.service_name == "yolo" and s.is_running for s in services),
                    moondream_instances=len([s for s in services if s.service_name == "moondream" and s.is_running]),
                    message_bus_connected=True,  # Placeholder
                    frames_processed=0,  # Would get from services
                    average_fps=0.0
                )
                
                return SystemStatusResponse(
                    system_status=system_status,
                    services=services,
                    uptime_seconds=uptime
                )
                
            except Exception as e:
                log_error_with_context(self.logger, e, operation="get_system_status")
                raise HTTPException(status_code=500, detail="Failed to get system status")
        
        @self.app.post("/chat", response_model=dict)
        async def send_chat_message(request: ChatRequest):
            """Send a chat message to the VLM."""
            try:
                # This would integrate with the message bus to send chat requests
                # For now, return a placeholder response
                return {
                    "message": f"Received: {request.message}",
                    "status": "queued",
                    "frame_id": request.frame_id
                }
                
            except Exception as e:
                log_error_with_context(self.logger, e, operation="send_chat_message")
                raise HTTPException(status_code=500, detail="Failed to send chat message")
        
        @self.app.post("/camera/control", response_model=CameraControlResponse)
        async def camera_control(request: CameraControlRequest):
            """Control camera operations."""
            try:
                if not self.camera_service:
                    raise HTTPException(status_code=503, detail="Camera service not available")
                
                if request.action == "start":
                    await self.camera_service.start()
                    return CameraControlResponse(
                        success=True,
                        message="Camera started successfully"
                    )
                elif request.action == "stop":
                    await self.camera_service.stop()
                    return CameraControlResponse(
                        success=True,
                        message="Camera stopped successfully"
                    )
                elif request.action == "configure":
                    # Handle camera configuration
                    if request.parameters:
                        for param, value in request.parameters.items():
                            await self.camera_service.set_camera_parameter(param, value)
                    
                    camera_info = await self.camera_service.get_camera_info()
                    return CameraControlResponse(
                        success=True,
                        message="Camera configured successfully",
                        camera_info=camera_info
                    )
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
                
            except Exception as e:
                log_error_with_context(self.logger, e, operation="camera_control")
                return CameraControlResponse(
                    success=False,
                    message=f"Camera control failed: {str(e)}"
                )
        
        @self.app.post("/service/control", response_model=ServiceControlResponse)
        async def service_control(request: ServiceControlRequest):
            """Control service operations."""
            try:
                service_map = {
                    "camera": self.camera_service,
                    "yolo": self.yolo_service,
                    "moondream": self.moondream_service,
                    "fusion": self.fusion_service
                }
                
                service = service_map.get(request.service)
                if not service:
                    raise HTTPException(status_code=404, detail=f"Service not found: {request.service}")
                
                if request.action == "start":
                    await service.start()
                    message = f"{request.service} service started successfully"
                elif request.action == "stop":
                    await service.stop()
                    message = f"{request.service} service stopped successfully"
                elif request.action == "restart":
                    await service.stop()
                    await service.start()
                    message = f"{request.service} service restarted successfully"
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
                
                # Get service info
                service_info = {}
                if hasattr(service, 'get_service_info'):
                    service_info = await service.get_service_info()
                elif hasattr(service, 'get_camera_info'):
                    service_info = await service.get_camera_info()
                
                return ServiceControlResponse(
                    service=request.service,
                    action=request.action,
                    success=True,
                    message=message,
                    service_info=service_info
                )
                
            except Exception as e:
                log_error_with_context(self.logger, e, operation="service_control")
                return ServiceControlResponse(
                    service=request.service,
                    action=request.action,
                    success=False,
                    message=f"Service control failed: {str(e)}"
                )
        
        @self.app.get("/ui/settings", response_model=UIConfigResponse)
        async def get_ui_settings():
            """Get UI configuration settings."""
            return UIConfigResponse(
                settings=self.ui_settings,
                success=True,
                message="UI settings retrieved successfully"
            )
        
        @self.app.post("/ui/settings", response_model=UIConfigResponse)
        async def update_ui_settings(request: UIConfigRequest):
            """Update UI configuration settings."""
            try:
                self.ui_settings = request.settings
                
                return UIConfigResponse(
                    settings=self.ui_settings,
                    success=True,
                    message="UI settings updated successfully"
                )
                
            except Exception as e:
                log_error_with_context(self.logger, e, operation="update_ui_settings")
                return UIConfigResponse(
                    settings=self.ui_settings,
                    success=False,
                    message=f"Failed to update UI settings: {str(e)}"
                )
        
        @self.app.get("/metrics", response_model=PerformanceMetrics)
        async def get_performance_metrics():
            """Get performance metrics."""
            try:
                # Collect metrics from services
                camera_fps = 0.0
                yolo_fps = 0.0
                vlm_fps = 0.0
                fusion_fps = 0.0
                
                avg_processing_times = {}
                memory_usage = {}
                
                # This would be populated from actual service metrics
                return PerformanceMetrics(
                    camera_fps=camera_fps,
                    yolo_fps=yolo_fps,
                    vlm_fps=vlm_fps,
                    fusion_fps=fusion_fps,
                    avg_processing_times=avg_processing_times,
                    memory_usage=memory_usage
                )
                
            except Exception as e:
                log_error_with_context(self.logger, e, operation="get_performance_metrics")
                raise HTTPException(status_code=500, detail="Failed to get performance metrics")
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """WebSocket endpoint for real-time communication."""
            try:
                await self.websocket_handler.handle_connection(websocket, client_id)
            except Exception as e:
                log_error_with_context(
                    self.logger,
                    e,
                    {"client_id": client_id},
                    "websocket_endpoint"
                )
        
        # Error handlers
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error="HTTP Exception",
                    message=str(exc.detail)
                ).dict()
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request, exc):
            log_error_with_context(
                self.logger,
                exc,
                {"url": str(request.url), "method": request.method},
                "general_exception"
            )
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="Internal Server Error",
                    message="An unexpected error occurred"
                ).dict()
            )
    
    async def start(self):
        """Start the API and WebSocket handler."""
        try:
            await self.websocket_handler.start()
            self.logger.info(f"API server starting on {self.config.api.host}:{self.config.api.port}")
        except Exception as e:
            log_error_with_context(self.logger, e, operation="api_start")
            raise
    
    async def stop(self):
        """Stop the API and WebSocket handler."""
        try:
            await self.websocket_handler.stop()
            self.logger.info("API server stopped")
        except Exception as e:
            log_error_with_context(self.logger, e, operation="api_stop")
    
    def set_services(self, camera=None, yolo=None, moondream=None, fusion=None):
        """Set service references for the API."""
        self.camera_service = camera
        self.yolo_service = yolo
        self.moondream_service = moondream
        self.fusion_service = fusion


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    api = MoondreamAPI()
    return api.app


async def run_api_server():
    """Run the API server."""
    config = get_config()
    logger = setup_logging("api_server")
    
    api = MoondreamAPI()
    
    try:
        await api.start()
        
        # Run the server
        config_obj = uvicorn.Config(
            api.app,
            host=config.api.host,
            port=config.api.port,
            log_level="info",
            access_log=False  # We handle logging ourselves
        )
        
        server = uvicorn.Server(config_obj)
        await server.serve()
        
    except Exception as e:
        log_error_with_context(logger, e, operation="run_api_server")
        raise
    finally:
        await api.stop()


if __name__ == "__main__":
    asyncio.run(run_api_server())
