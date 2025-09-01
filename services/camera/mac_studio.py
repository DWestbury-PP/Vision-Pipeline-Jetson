"""Mac Studio Display camera implementation using OpenCV with AVFoundation."""

import cv2
import numpy as np
import asyncio
import time
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

from .base import StreamingCameraInterface, CameraFactory
from ..shared.models import FrameMetadata
from ..shared.logging_config import setup_logging, log_error_with_context, log_performance_metrics
from ..shared.config import get_config


class MacStudioCamera(StreamingCameraInterface):
    """Mac Studio Display camera implementation."""
    
    def __init__(self, camera_id: str = "mac_studio"):
        super().__init__(camera_id)
        self.logger = setup_logging("mac_studio_camera")
        self.config = get_config()
        self.cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time = 0
        self._target_fps = self.config.camera.fps
        self._frame_interval = 1.0 / self._target_fps
        
        # Camera settings
        self.width = self.config.camera.width
        self.height = self.config.camera.height
        self.camera_index = self.config.camera.index
        
    async def initialize(self, **kwargs) -> bool:
        """Initialize the Mac Studio Display camera.
        
        Args:
            **kwargs: Additional initialization parameters
                - width: Frame width (default from config)
                - height: Frame height (default from config)
                - fps: Target FPS (default from config)
                - index: Camera index (default from config)
        """
        try:
            # Override defaults with provided kwargs
            self.width = kwargs.get('width', self.width)
            self.height = kwargs.get('height', self.height)
            self._target_fps = kwargs.get('fps', self._target_fps)
            self.camera_index = kwargs.get('index', self.camera_index)
            self._frame_interval = 1.0 / self._target_fps
            
            # Initialize camera with AVFoundation backend (Mac optimized)
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera at index {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self._target_fps)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(
                f"Camera initialized successfully",
                extra={
                    "camera_index": self.camera_index,
                    "requested_resolution": f"{self.width}x{self.height}",
                    "actual_resolution": f"{actual_width}x{actual_height}",
                    "requested_fps": self._target_fps,
                    "actual_fps": actual_fps
                }
            )
            
            self.is_active = True
            return True
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"camera_index": self.camera_index},
                "camera_initialization"
            )
            return False
    
    async def start_capture(self) -> None:
        """Start camera capture."""
        if not self.is_active:
            raise RuntimeError("Camera not initialized")
            
        self.logger.info("Camera capture started")
    
    async def stop_capture(self) -> None:
        """Stop camera capture."""
        self.logger.info("Camera capture stopped")
    
    async def capture_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Capture a single frame from the Mac Studio Display camera."""
        if not self.cap or not self.is_active:
            return None
            
        try:
            start_time = time.perf_counter()
            
            # Rate limiting to target FPS
            current_time = time.perf_counter()
            time_since_last = current_time - self._last_frame_time
            if time_since_last < self._frame_interval:
                await asyncio.sleep(self._frame_interval - time_since_last)
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.warning("Failed to capture frame")
                return None
            
            capture_time = time.perf_counter()
            self._last_frame_time = capture_time
            
            # Create metadata
            frame_id = self.get_next_frame_id()
            metadata = FrameMetadata(
                frame_id=frame_id,
                timestamp=datetime.utcnow(),
                width=frame.shape[1],
                height=frame.shape[0],
                channels=frame.shape[2] if len(frame.shape) > 2 else 1,
                fps=self._target_fps,
                camera_id=self.camera_id
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Log performance metrics
            if self.config.pipeline.enable_performance_metrics and frame_id % 30 == 0:
                log_performance_metrics(
                    self.logger,
                    "frame_capture",
                    processing_time,
                    frame_id=frame_id,
                    model_name="mac_studio_camera",
                    frame_size=frame.nbytes,
                    resolution=f"{metadata.width}x{metadata.height}"
                )
            
            return frame, metadata
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"frame_id": getattr(self, '_frame_count', 0)},
                "frame_capture"
            )
            return None
    
    async def release(self) -> None:
        """Release camera resources."""
        try:
            await super().release()
            
            if self.cap:
                self.cap.release()
                self.cap = None
                
            self.is_active = False
            self.logger.info("Camera resources released")
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="camera_release")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get Mac Studio Display camera capabilities."""
        if not self.cap:
            return {}
            
        try:
            return {
                "supported_resolutions": [
                    (1920, 1080),
                    (1280, 720),
                    (640, 480)
                ],
                "max_fps": 60,
                "color_formats": ["BGR", "RGB"],
                "features": {
                    "auto_focus": True,
                    "auto_exposure": True,
                    "auto_white_balance": True
                },
                "current_settings": {
                    "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": self.cap.get(cv2.CAP_PROP_FPS),
                    "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                    "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
                    "saturation": self.cap.get(cv2.CAP_PROP_SATURATION),
                    "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE)
                }
            }
        except Exception as e:
            log_error_with_context(self.logger, e, operation="get_capabilities")
            return {}
    
    def set_parameter(self, parameter: str, value: Any) -> bool:
        """Set a camera parameter."""
        if not self.cap:
            return False
            
        try:
            param_map = {
                "width": cv2.CAP_PROP_FRAME_WIDTH,
                "height": cv2.CAP_PROP_FRAME_HEIGHT,
                "fps": cv2.CAP_PROP_FPS,
                "brightness": cv2.CAP_PROP_BRIGHTNESS,
                "contrast": cv2.CAP_PROP_CONTRAST,
                "saturation": cv2.CAP_PROP_SATURATION,
                "exposure": cv2.CAP_PROP_EXPOSURE,
                "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
                "buffer_size": cv2.CAP_PROP_BUFFERSIZE
            }
            
            if parameter not in param_map:
                self.logger.warning(f"Unsupported parameter: {parameter}")
                return False
            
            cv_param = param_map[parameter]
            result = self.cap.set(cv_param, value)
            
            self.logger.info(
                f"Set camera parameter {parameter} = {value}",
                extra={"parameter": parameter, "value": value, "success": result}
            )
            
            return result
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"parameter": parameter, "value": value},
                "set_parameter"
            )
            return False
    
    def get_parameter(self, parameter: str) -> Any:
        """Get a camera parameter value."""
        if not self.cap:
            return None
            
        try:
            param_map = {
                "width": cv2.CAP_PROP_FRAME_WIDTH,
                "height": cv2.CAP_PROP_FRAME_HEIGHT,
                "fps": cv2.CAP_PROP_FPS,
                "brightness": cv2.CAP_PROP_BRIGHTNESS,
                "contrast": cv2.CAP_PROP_CONTRAST,
                "saturation": cv2.CAP_PROP_SATURATION,
                "exposure": cv2.CAP_PROP_EXPOSURE,
                "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
                "buffer_size": cv2.CAP_PROP_BUFFERSIZE
            }
            
            if parameter not in param_map:
                return None
            
            cv_param = param_map[parameter]
            return self.cap.get(cv_param)
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"parameter": parameter},
                "get_parameter"
            )
            return None


# Register the Mac Studio camera with the factory
CameraFactory.register_camera("mac_studio", MacStudioCamera)
