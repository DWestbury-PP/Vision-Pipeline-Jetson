"""Pydantic models for API requests and responses."""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime

from ..shared.models import (
    BoundingBox, DetectionResult, VLMResult, ProcessingPipelineResult,
    SystemStatus, UISettings, FrameMetadata, ChatMessage, ChatResponse
)


# WebSocket message types
class WSMessage(BaseModel):
    """Base WebSocket message."""
    message_type: str = Field(..., description="Type of WebSocket message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSFrameUpdate(WSMessage):
    """WebSocket message for frame updates."""
    message_type: Literal["frame_update"] = Field(default="frame_update")
    frame_metadata: FrameMetadata
    frame_data_url: Optional[str] = Field(None, description="Base64 encoded frame data URL")


class WSDetectionUpdate(WSMessage):
    """WebSocket message for detection updates."""
    message_type: Literal["detection_update"] = Field(default="detection_update")
    frame_id: int
    yolo_result: Optional[DetectionResult] = None
    vlm_result: Optional[VLMResult] = None
    fused_result: Optional[ProcessingPipelineResult] = None


class WSStatusUpdate(WSMessage):
    """WebSocket message for status updates."""
    message_type: Literal["status_update"] = Field(default="status_update")
    status: SystemStatus


class WSChatResponse(WSMessage):
    """WebSocket message for chat responses."""
    message_type: Literal["chat_response"] = Field(default="chat_response")
    chat_response: ChatResponse


class WSError(WSMessage):
    """WebSocket error message."""
    message_type: Literal["error"] = Field(default="error")
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None


# REST API request/response models
class ServiceStatusResponse(BaseModel):
    """Response for service status."""
    service_name: str
    is_running: bool
    details: Dict[str, Any]


class SystemStatusResponse(BaseModel):
    """Response for system status."""
    system_status: SystemStatus
    services: List[ServiceStatusResponse]
    uptime_seconds: float


class ConfigurationUpdateRequest(BaseModel):
    """Request to update configuration."""
    service: Optional[str] = Field(None, description="Specific service to update (optional)")
    parameters: Dict[str, Any] = Field(..., description="Parameters to update")


class ConfigurationResponse(BaseModel):
    """Response for configuration."""
    current_config: Dict[str, Any]
    updated_parameters: Optional[Dict[str, Any]] = None
    success: bool
    message: str


class ChatRequest(BaseModel):
    """Request for chat interaction."""
    message: str = Field(..., description="User message")
    frame_id: Optional[int] = Field(None, description="Frame to use as context")
    include_current_frame: bool = Field(True, description="Include current camera frame")


class CameraControlRequest(BaseModel):
    """Request for camera control."""
    action: str = Field(..., description="Action: start, stop, configure")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Action parameters")


class CameraControlResponse(BaseModel):
    """Response for camera control."""
    success: bool
    message: str
    camera_info: Optional[Dict[str, Any]] = None


class ServiceControlRequest(BaseModel):
    """Request for service control."""
    service: str = Field(..., description="Service name")
    action: str = Field(..., description="Action: start, stop, restart, configure")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Action parameters")


class ServiceControlResponse(BaseModel):
    """Response for service control."""
    service: str
    action: str
    success: bool
    message: str
    service_info: Optional[Dict[str, Any]] = None


class FrameRequest(BaseModel):
    """Request for frame data."""
    frame_id: Optional[int] = Field(None, description="Specific frame ID")
    include_detections: bool = Field(True, description="Include detection overlays")
    include_labels: bool = Field(True, description="Include detection labels")
    format: str = Field(default="jpeg", description="Image format: jpeg, png")
    quality: int = Field(default=85, ge=1, le=100, description="JPEG quality")


class FrameResponse(BaseModel):
    """Response for frame data."""
    frame_metadata: FrameMetadata
    frame_data_url: str = Field(..., description="Base64 encoded image data URL")
    detections: Optional[DetectionResult] = None
    vlm_result: Optional[VLMResult] = None


class PerformanceMetrics(BaseModel):
    """Performance metrics response."""
    camera_fps: float
    yolo_fps: float
    vlm_fps: float
    fusion_fps: float
    avg_processing_times: Dict[str, float]
    memory_usage: Dict[str, float]
    gpu_utilization: Optional[float] = None


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall health status")
    services: Dict[str, str] = Field(..., description="Individual service health")
    message_bus: str = Field(..., description="Message bus health")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime_seconds: float


# UI-specific models
class UIConfigRequest(BaseModel):
    """Request to update UI configuration."""
    settings: UISettings


class UIConfigResponse(BaseModel):
    """Response for UI configuration."""
    settings: UISettings
    success: bool
    message: str


class AnnotationRequest(BaseModel):
    """Request for image annotation."""
    frame_id: int
    bounding_boxes: List[BoundingBox]
    show_confidence: bool = True
    show_labels: bool = True
    box_color: str = "#00FF00"
    text_color: str = "#FFFFFF"
    line_thickness: int = 2


class AnnotationResponse(BaseModel):
    """Response for image annotation."""
    annotated_frame_url: str = Field(..., description="Base64 encoded annotated image")
    annotation_count: int


# Error responses
class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Pagination models
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any]
    total_count: int
    page: int
    size: int
    total_pages: int
    has_next: bool
    has_previous: bool


# Batch operation models
class BatchDetectionRequest(BaseModel):
    """Request for batch detection processing."""
    frame_ids: List[int] = Field(..., description="List of frame IDs to process")
    yolo_enabled: bool = True
    vlm_enabled: bool = True
    return_images: bool = False


class BatchDetectionResponse(BaseModel):
    """Response for batch detection processing."""
    results: List[ProcessingPipelineResult]
    processing_time_ms: float
    success_count: int
    error_count: int
    errors: List[Dict[str, Any]] = Field(default_factory=list)
