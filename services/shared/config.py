"""Configuration management for the Moondream Vision Pipeline."""

import os
from typing import Optional, Literal
from pydantic import Field
from pydantic import BaseSettings


class RedisConfig(BaseSettings):
    """Redis configuration."""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    
    model_config = {"extra": "ignore"}


class CameraConfig(BaseSettings):
    """Camera configuration."""
    type: Literal["mac_studio", "jetson_csi", "usb", "mock"] = Field(default="mock", env="CAMERA_TYPE")
    index: int = Field(default=0, env="CAMERA_INDEX")
    width: int = Field(default=1920, env="CAMERA_WIDTH")
    height: int = Field(default=1080, env="CAMERA_HEIGHT")
    fps: int = Field(default=30, env="CAMERA_FPS")
    
    model_config = {"extra": "ignore"}


class YOLOConfig(BaseSettings):
    """YOLO configuration."""
    model: str = Field(default="yolo11n.pt", env="YOLO_MODEL")
    confidence: float = Field(default=0.5, env="YOLO_CONFIDENCE")
    device: str = Field(default="mps", env="YOLO_DEVICE")
    frame_stride: int = Field(default=1, env="YOLO_FRAME_STRIDE")
    
    model_config = {"extra": "ignore"}


class MoondreamConfig(BaseSettings):
    """Moondream VLM configuration."""
    model: str = Field(default="vikhyatk/moondream2", env="MOONDREAM_MODEL")
    device: str = Field(default="mps", env="MOONDREAM_DEVICE")
    instances: int = Field(default=2, env="MOONDREAM_INSTANCES")
    batch_size: int = Field(default=1, env="MOONDREAM_BATCH_SIZE")
    frame_stride: int = Field(default=10, env="VLM_FRAME_STRIDE")
    
    model_config = {"extra": "ignore"}


class PipelineConfig(BaseSettings):
    """Pipeline processing configuration."""
    enable_scene_change_detection: bool = Field(default=True, env="ENABLE_SCENE_CHANGE_DETECTION")
    scene_change_threshold: float = Field(default=0.3, env="SCENE_CHANGE_THRESHOLD")
    max_queue_size: int = Field(default=10, env="MAX_QUEUE_SIZE")
    ui_ring_buffer_size: int = Field(default=3, env="UI_RING_BUFFER_SIZE")
    enable_performance_metrics: bool = Field(default=True, env="ENABLE_PERFORMANCE_METRICS")
    
    model_config = {"extra": "ignore"}


class APIConfig(BaseSettings):
    """API server configuration."""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    websocket_port: int = Field(default=8001, env="WEBSOCKET_PORT")
    
    model_config = {"extra": "ignore"}


class FrontendConfig(BaseSettings):
    """Frontend configuration."""
    port: int = Field(default=3000, env="FRONTEND_PORT")
    api_base_url: str = Field(default="http://localhost:8000", env="API_BASE_URL")
    
    model_config = {"extra": "ignore"}


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: Literal["json", "text"] = Field(default="json", env="LOG_FORMAT")
    
    model_config = {"extra": "ignore"}


class Config(BaseSettings):
    """Main configuration class with flattened configuration."""
    
    # Redis configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Camera configuration
    camera_type: Literal["mac_studio", "jetson_csi", "usb", "mock"] = Field(default="mock", env="CAMERA_TYPE")
    camera_index: int = Field(default=0, env="CAMERA_INDEX")
    camera_width: int = Field(default=1920, env="CAMERA_WIDTH")
    camera_height: int = Field(default=1080, env="CAMERA_HEIGHT")
    camera_fps: int = Field(default=30, env="CAMERA_FPS")
    csi_sensor_mode: int = Field(default=0, env="CSI_SENSOR_MODE")
    
    # YOLO configuration
    yolo_model: str = Field(default="yolo11n.pt", env="YOLO_MODEL")
    yolo_confidence: float = Field(default=0.01, env="YOLO_CONFIDENCE")  # Very low threshold for mock objects
    yolo_device: str = Field(default="cuda", env="YOLO_DEVICE")
    yolo_frame_stride: int = Field(default=1, env="YOLO_FRAME_STRIDE")
    
    # Moondream VLM configuration
    moondream_model: str = Field(default="vikhyatk/moondream2", env="MOONDREAM_MODEL")
    moondream_device: str = Field(default="cuda", env="MOONDREAM_DEVICE")
    moondream_instances: int = Field(default=2, env="MOONDREAM_INSTANCES")
    moondream_batch_size: int = Field(default=1, env="MOONDREAM_BATCH_SIZE")
    vlm_frame_stride: int = Field(default=10, env="VLM_FRAME_STRIDE")
    
    # Pipeline processing configuration
    enable_scene_change_detection: bool = Field(default=True, env="ENABLE_SCENE_CHANGE_DETECTION")
    scene_change_threshold: float = Field(default=0.3, env="SCENE_CHANGE_THRESHOLD")
    max_queue_size: int = Field(default=10, env="MAX_QUEUE_SIZE")
    ui_ring_buffer_size: int = Field(default=3, env="UI_RING_BUFFER_SIZE")
    enable_performance_metrics: bool = Field(default=True, env="ENABLE_PERFORMANCE_METRICS")
    
    # API server configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    websocket_port: int = Field(default=8001, env="WEBSOCKET_PORT")
    
    # Frontend configuration
    frontend_port: int = Field(default=3000, env="FRONTEND_PORT")
    api_base_url: str = Field(default="http://localhost:8000", env="API_BASE_URL")
    
    # Logging configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: Literal["json", "text"] = Field(default="json", env="LOG_FORMAT")
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment."""
    global config
    config = Config()
    return config
