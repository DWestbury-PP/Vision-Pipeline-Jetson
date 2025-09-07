"""Jetson CSI camera implementation for IMX219-83 Stereo Binocular Camera using GStreamer."""

import cv2
import numpy as np
import asyncio
import time
import subprocess
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

from .base import StreamingCameraInterface, CameraFactory
from ..shared.models import FrameMetadata
from ..shared.logging_config import setup_logging, log_error_with_context, log_performance_metrics
from ..shared.config import get_config


class JetsonCSICamera(StreamingCameraInterface):
    """Jetson CSI camera implementation for IMX219-83 Stereo Binocular Camera."""
    
    def __init__(self, camera_id: str = "jetson_csi"):
        super().__init__(camera_id)
        self.logger = setup_logging("jetson_csi_camera")
        self.config = get_config()
        self.cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time = 0
        self._target_fps = self.config.camera_fps
        self._frame_interval = 1.0 / self._target_fps
        
        # Camera settings
        self.width = self.config.camera_width
        self.height = self.config.camera_height
        self.camera_index = self.config.camera_index
        self.sensor_mode = getattr(self.config, 'csi_sensor_mode', 0)  # Default sensor mode
        
        # GStreamer pipeline for IMX219-83
        self.gst_pipeline = None
        
    def _build_gstreamer_pipeline(self) -> str:
        """Build GStreamer pipeline for IMX219-83 Stereo Binocular Camera."""
        # IMX219-83 supports various modes, mode 0 is typically 1920x1080@30fps
        pipeline = (
            f"nvarguscamerasrc sensor-id={self.camera_index} sensor-mode={self.sensor_mode} ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate={self._target_fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink max-buffers=1 drop=true"
        )
        return pipeline
        
    async def initialize(self, **kwargs) -> bool:
        """Initialize the Jetson CSI camera.
        
        Args:
            **kwargs: Additional initialization parameters
                - width: Frame width (default from config)
                - height: Frame height (default from config)
                - fps: Target FPS (default from config)
                - index: Camera index/sensor-id (default from config)
                - sensor_mode: CSI sensor mode (default 0)
        """
        try:
            # Override defaults with provided kwargs
            self.width = kwargs.get('width', self.width)
            self.height = kwargs.get('height', self.height)
            self._target_fps = kwargs.get('fps', self._target_fps)
            self.camera_index = kwargs.get('index', self.camera_index)
            self.sensor_mode = kwargs.get('sensor_mode', self.sensor_mode)
            self._frame_interval = 1.0 / self._target_fps
            
            # Check if camera is available
            if not await self._check_camera_availability():
                self.logger.error(f"CSI camera sensor-id {self.camera_index} not available")
                return False
            
            # Build GStreamer pipeline
            self.gst_pipeline = self._build_gstreamer_pipeline()
            self.logger.info(f"GStreamer pipeline: {self.gst_pipeline}")
            
            # Initialize camera with GStreamer pipeline
            self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open CSI camera with sensor-id {self.camera_index}")
                return False
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(
                f"CSI camera initialized successfully",
                extra={
                    "sensor_id": self.camera_index,
                    "sensor_mode": self.sensor_mode,
                    "requested_resolution": f"{self.width}x{self.height}",
                    "actual_resolution": f"{actual_width}x{actual_height}",
                    "requested_fps": self._target_fps,
                    "actual_fps": actual_fps,
                    "gst_pipeline": self.gst_pipeline
                }
            )
            
            self.is_active = True
            return True
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"sensor_id": self.camera_index, "sensor_mode": self.sensor_mode},
                "csi_camera_initialization"
            )
            return False
    
    async def _check_camera_availability(self) -> bool:
        """Check if the CSI camera is available."""
        try:
            # Use v4l2-ctl to check available cameras
            result = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                self.logger.debug(f"Available cameras:\n{result.stdout}")
                return True
            else:
                # Try alternative method - check if argus daemon is running
                result = subprocess.run(
                    ["pgrep", "-f", "nvargus-daemon"],
                    capture_output=True,
                    timeout=5
                )
                return result.returncode == 0
                
        except subprocess.TimeoutExpired:
            self.logger.warning("Camera availability check timed out")
            return True  # Assume available if check fails
        except Exception as e:
            log_error_with_context(self.logger, e, operation="camera_availability_check")
            return True  # Assume available if check fails
    
    async def start_capture(self) -> None:
        """Start camera capture."""
        if not self.is_active:
            raise RuntimeError("Camera not initialized")
            
        self.logger.info("CSI camera capture started")
    
    async def stop_capture(self) -> None:
        """Stop camera capture."""
        self.logger.info("CSI camera capture stopped")
    
    async def capture_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Capture a single frame from the Jetson CSI camera."""
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
                self.logger.warning("Failed to capture frame from CSI camera")
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
            if self.config.enable_performance_metrics and frame_id % 30 == 0:
                log_performance_metrics(
                    self.logger,
                    "frame_capture",
                    processing_time,
                    frame_id=frame_id,
                    model_name="jetson_csi_camera",
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
            self.logger.info("CSI camera resources released")
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="camera_release")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get Jetson CSI camera capabilities."""
        if not self.cap:
            return {}
            
        try:
            return {
                "supported_resolutions": [
                    (3280, 2464),  # Mode 0: Full resolution
                    (1920, 1080),  # Mode 1: 1080p
                    (1640, 1232),  # Mode 2: 4:3 aspect ratio
                    (1280, 720),   # Mode 3: 720p
                ],
                "supported_sensor_modes": [0, 1, 2, 3],
                "max_fps": 30,
                "color_formats": ["BGR", "NV12"],
                "features": {
                    "auto_focus": True,
                    "auto_exposure": True,
                    "auto_white_balance": True,
                    "stereo_support": True,  # IMX219-83 Stereo Binocular
                    "hardware_acceleration": True
                },
                "current_settings": {
                    "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": self.cap.get(cv2.CAP_PROP_FPS),
                    "sensor_mode": self.sensor_mode,
                    "sensor_id": self.camera_index
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
            # Map of supported parameters for CSI camera
            param_map = {
                "sensor_mode": "sensor_mode",  # Special handling needed
                "exposure_compensation": "exposure_compensation",
                "ae_lock": "ae_lock",
                "awb_lock": "awb_lock",
                "saturation": "saturation",
            }
            
            if parameter == "sensor_mode":
                # Sensor mode requires camera reinitialization
                self.logger.warning("Sensor mode change requires camera reinitialization")
                return False
            
            if parameter not in param_map:
                self.logger.warning(f"Unsupported parameter for CSI camera: {parameter}")
                return False
            
            # Note: Most CSI camera parameters are controlled via GStreamer pipeline
            # and cannot be changed at runtime with OpenCV
            self.logger.info(
                f"CSI camera parameter {parameter} = {value} (note: limited runtime control)",
                extra={"parameter": parameter, "value": value}
            )
            
            return True
            
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
            if parameter == "sensor_mode":
                return self.sensor_mode
            elif parameter == "sensor_id":
                return self.camera_index
            elif parameter == "gst_pipeline":
                return self.gst_pipeline
            else:
                # Most CSI parameters are not accessible via OpenCV
                return None
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"parameter": parameter},
                "get_parameter"
            )
            return None


# Register the Jetson CSI camera with the factory
CameraFactory.register_camera("jetson_csi", JetsonCSICamera)
