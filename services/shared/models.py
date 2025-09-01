"""Shared data models for the Moondream Vision Pipeline."""

from typing import List, Optional, Dict, Any, Tuple, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np


class BoundingBox(BaseModel):
    """Bounding box representation."""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate") 
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_name: str = Field(..., description="Object class name")
    class_id: Optional[int] = Field(None, description="Object class ID")
    
    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def center(self) -> Tuple[float, float]:
        """Get bounding box center point."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Point(BaseModel):
    """2D point representation."""
    x: float
    y: float
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class DetectionResult(BaseModel):
    """Object detection result."""
    bounding_boxes: List[BoundingBox] = Field(default_factory=list)
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_name: str = Field(..., description="Name of the detection model")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VLMResult(BaseModel):
    """VLM processing result."""
    caption: Optional[str] = Field(None, description="Image caption")
    description: Optional[str] = Field(None, description="Detailed description")
    objects: List[str] = Field(default_factory=list, description="Detected objects")
    bounding_boxes: List[BoundingBox] = Field(default_factory=list)
    points: List[Point] = Field(default_factory=list)
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_name: str = Field(..., description="Name of the VLM model")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class ChatMessage(BaseModel):
    """Chat message for VLM interaction."""
    message: str = Field(..., description="User message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    frame_id: Optional[int] = Field(None, description="Associated frame ID")


class ChatResponse(BaseModel):
    """Chat response from VLM."""
    response: str = Field(..., description="VLM response")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_name: str = Field(..., description="Name of the VLM model")
    frame_id: Optional[int] = Field(None, description="Associated frame ID")


class FrameMetadata(BaseModel):
    """Metadata for captured frames."""
    frame_id: int = Field(..., description="Unique frame identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    width: int = Field(..., description="Frame width in pixels")
    height: int = Field(..., description="Frame height in pixels")
    channels: int = Field(default=3, description="Number of color channels")
    fps: float = Field(..., description="Capture frame rate")
    camera_id: str = Field(..., description="Camera identifier")
    file_path: Optional[str] = Field(None, description="Temporary file path if saved")


class FrameData(BaseModel):
    """Complete frame data with metadata."""
    metadata: FrameMetadata
    # Note: actual image data is handled separately due to serialization constraints
    
    class Config:
        arbitrary_types_allowed = True


class ProcessingPipelineResult(BaseModel):
    """Combined result from the processing pipeline."""
    frame_metadata: FrameMetadata
    yolo_result: Optional[DetectionResult] = None
    vlm_result: Optional[VLMResult] = None
    fusion_timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_processing_time_ms: float = Field(..., description="Total pipeline processing time")


class SystemStatus(BaseModel):
    """System status information."""
    camera_active: bool = Field(..., description="Camera capture status")
    yolo_active: bool = Field(..., description="YOLO service status")
    moondream_instances: int = Field(..., description="Number of active Moondream instances")
    message_bus_connected: bool = Field(..., description="Message bus connection status")
    frames_processed: int = Field(default=0, description="Total frames processed")
    average_fps: float = Field(default=0.0, description="Average processing FPS")
    memory_usage_mb: float = Field(default=0.0, description="Memory usage in MB")
    gpu_utilization: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU utilization percentage")


class UISettings(BaseModel):
    """UI configuration settings."""
    show_bounding_boxes: bool = Field(default=True)
    show_confidence: bool = Field(default=True)
    show_labels: bool = Field(default=True)
    mirror_mode: bool = Field(default=False)
    yolo_enabled: bool = Field(default=True)
    vlm_enabled: bool = Field(default=True)
    vlm_frame_stride: int = Field(default=10, ge=1, le=100)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ServiceConfiguration(BaseModel):
    """Service-level configuration."""
    service_name: str = Field(..., description="Name of the service")
    model_path: Optional[str] = Field(None, description="Path to model file")
    device: str = Field(default="cpu", description="Compute device")
    batch_size: int = Field(default=1, ge=1, description="Batch size for processing")
    max_queue_size: int = Field(default=10, ge=1, description="Maximum queue size")
    enabled: bool = Field(default=True, description="Service enabled status")


# Message bus message types
class BusMessage(BaseModel):
    """Base message for the message bus."""
    message_type: str = Field(..., description="Type of message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_service: str = Field(..., description="Service that sent the message")


class FrameMessage(BusMessage):
    """Frame message for the message bus."""
    message_type: Literal["frame"] = Field(default="frame")
    frame_metadata: FrameMetadata
    # Frame data is transmitted separately


class DetectionMessage(BusMessage):
    """Detection result message."""
    message_type: Literal["detection"] = Field(default="detection")
    result: DetectionResult
    frame_id: int


class VLMMessage(BusMessage):
    """VLM result message."""
    message_type: Literal["vlm"] = Field(default="vlm")
    result: VLMResult
    frame_id: int


class ChatRequestMessage(BusMessage):
    """Chat request message."""
    message_type: Literal["chat_request"] = Field(default="chat_request")
    chat_message: ChatMessage


class ChatResponseMessage(BusMessage):
    """Chat response message."""
    message_type: Literal["chat_response"] = Field(default="chat_response")
    chat_response: ChatResponse


class StatusMessage(BusMessage):
    """System status message."""
    message_type: Literal["status"] = Field(default="status")
    status: SystemStatus


class ConfigurationMessage(BusMessage):
    """Configuration update message.""" 
    message_type: Literal["configuration"] = Field(default="configuration")
    configuration: Dict[str, Any]
