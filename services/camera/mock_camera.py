"""Mock camera implementation for testing without OpenCV dependencies."""

import numpy as np
import asyncio
import time
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

from .base import StreamingCameraInterface, CameraFactory
from ..shared.models import FrameMetadata
from ..shared.logging_config import setup_logging, log_error_with_context


class MockCamera(StreamingCameraInterface):
    """Mock camera implementation that generates synthetic frames."""
    
    def __init__(self, camera_id: str = "mock_camera"):
        super().__init__(camera_id)
        self.logger = setup_logging("mock_camera")
        self._frame_counter = 0
        self._is_capturing = False
        
        # Default settings
        self.width = 640
        self.height = 480
        self.fps = 30
        
    async def initialize(self, **kwargs) -> bool:
        """Initialize the mock camera."""
        try:
            self.width = kwargs.get('width', 640)
            self.height = kwargs.get('height', 480)
            self.fps = kwargs.get('fps', 30)
            
            self.logger.info(
                f"Mock camera initialized",
                extra={
                    "width": self.width,
                    "height": self.height,
                    "fps": self.fps
                }
            )
            return True
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="mock_camera_initialize")
            return False
    
    async def start_capture(self) -> bool:
        """Start capturing frames."""
        try:
            self._is_capturing = True
            self._frame_counter = 0
            self.logger.info("Mock camera capture started")
            return True
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="start_capture")
            return False
    
    async def capture_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Capture a synthetic frame."""
        try:
            if not self._is_capturing:
                return None
            
            # Generate a synthetic frame (gradient pattern)
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Create a simple gradient pattern that changes over time
            for y in range(self.height):
                for x in range(self.width):
                    # Create a time-varying pattern
                    t = self._frame_counter * 0.1
                    r = int((np.sin(t + x * 0.01) + 1) * 127)
                    g = int((np.cos(t + y * 0.01) + 1) * 127)
                    b = int((np.sin(t + (x + y) * 0.005) + 1) * 127)
                    frame[y, x] = [b, g, r]  # BGR format
            
            # Add frame counter text simulation (just a colored rectangle)
            text_height = 30
            text_width = 200
            if self.height > text_height and self.width > text_width:
                # Add a colored rectangle to simulate text
                color_intensity = (self._frame_counter % 255)
                frame[10:10+text_height, 10:10+text_width] = [color_intensity, 255-color_intensity, 128]
            
            # Create metadata
            timestamp = datetime.now()
            metadata = FrameMetadata(
                frame_id=self._frame_counter,
                timestamp=timestamp,
                width=self.width,
                height=self.height,
                channels=3,
                source_camera_id=self.camera_id,
                processing_metadata={}
            )
            
            self._frame_counter += 1
            
            # Simulate frame rate limiting
            await asyncio.sleep(1.0 / self.fps)
            
            return frame, metadata
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="capture_frame")
            return None
    
    async def stop_capture(self) -> bool:
        """Stop capturing frames."""
        try:
            self._is_capturing = False
            self.logger.info(f"Mock camera capture stopped. Frames captured: {self._frame_counter}")
            return True
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="stop_capture")
            return False
    
    async def release(self) -> bool:
        """Release camera resources."""
        try:
            self._is_capturing = False
            self.logger.info("Mock camera released")
            return True
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="release")
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get camera capabilities."""
        return {
            "resolutions": [
                {"width": 640, "height": 480},
                {"width": 1280, "height": 720},
                {"width": 1920, "height": 1080}
            ],
            "fps_ranges": [
                {"min": 1, "max": 60}
            ],
            "formats": ["BGR24"],
            "features": {
                "auto_exposure": False,
                "auto_white_balance": False,
                "synthetic_frames": True,
                "hardware_acceleration": False
            },
            "current_settings": {
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "format": "BGR24"
            }
        }
    
    def set_parameter(self, parameter: str, value: Any) -> bool:
        """Set a camera parameter."""
        try:
            if parameter == "width":
                self.width = int(value)
                return True
            elif parameter == "height":
                self.height = int(value)
                return True
            elif parameter == "fps":
                self.fps = float(value)
                return True
            else:
                self.logger.warning(f"Unknown parameter: {parameter}")
                return False
                
        except Exception as e:
            log_error_with_context(self.logger, e, {"parameter": parameter, "value": value}, "set_parameter")
            return False
    
    def get_parameter(self, parameter: str) -> Any:
        """Get a camera parameter value."""
        try:
            if parameter == "width":
                return self.width
            elif parameter == "height":
                return self.height
            elif parameter == "fps":
                return self.fps
            else:
                self.logger.warning(f"Unknown parameter: {parameter}")
                return None
                
        except Exception as e:
            log_error_with_context(self.logger, e, {"parameter": parameter}, "get_parameter")
            return None


# Register the mock camera with the factory
CameraFactory.register_camera("mock", MockCamera)
