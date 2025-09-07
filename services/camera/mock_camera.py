"""Mock camera implementation for testing the vision pipeline."""

import asyncio
import time
import numpy as np
from typing import Tuple, Optional, Dict, Any

from .base import CameraInterface, CameraFactory
from ..shared.models import FrameMetadata


class MockCamera(CameraInterface):
    """Mock camera that generates synthetic frames for testing."""
    
    def __init__(self, camera_id: str = "mock_camera"):
        super().__init__(camera_id)
        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.frame_interval = 1.0 / self.fps
        self.last_frame_time = 0
        
        # Mock camera parameters
        self.brightness = 50
        self.contrast = 50
        self.saturation = 50
        
    async def initialize(self, **kwargs) -> bool:
        """Initialize the mock camera."""
        try:
            self.width = kwargs.get('width', 1920)
            self.height = kwargs.get('height', 1080)
            self.fps = kwargs.get('fps', 30)
            self.frame_interval = 1.0 / self.fps
            
            print(f"🎥 Mock camera initialized: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"❌ Mock camera initialization failed: {e}")
            return False
    
    async def start_capture(self) -> None:
        """Start mock camera capture."""
        self.is_active = True
        self.reset_frame_count()
        self.last_frame_time = time.time()
        print("📹 Mock camera capture started")
    
    async def stop_capture(self) -> None:
        """Stop mock camera capture."""
        self.is_active = False
        print("⏹️ Mock camera capture stopped")
    
    async def capture_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Capture a synthetic frame."""
        if not self.is_active:
            return None
        
        # Throttle frame rate
        current_time = time.time()
        time_since_last = current_time - self.last_frame_time
        if time_since_last < self.frame_interval:
            await asyncio.sleep(self.frame_interval - time_since_last)
            current_time = time.time()
        
        self.last_frame_time = current_time
        
        try:
            # Generate synthetic frame with moving pattern
            frame = self._generate_synthetic_frame()
            
            # Create frame metadata
            metadata = FrameMetadata(
                frame_id=self.get_next_frame_id(),
                timestamp=current_time,
                camera_id=self.camera_id,
                width=self.width,
                height=self.height,
                format="BGR",
                source="mock_camera"
            )
            
            return frame, metadata
            
        except Exception as e:
            print(f"❌ Mock camera frame generation failed: {e}")
            return None
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate a synthetic frame with animated patterns."""
        # Create base frame with gradient
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add time-based animation
        t = time.time() * 2  # Animation speed
        
        # Create moving gradient
        for y in range(self.height):
            for x in range(self.width):
                # Create moving wave pattern
                wave_x = int(128 + 100 * np.sin(x * 0.01 + t))
                wave_y = int(128 + 100 * np.cos(y * 0.01 + t * 0.7))
                
                # RGB channels with different patterns
                frame[y, x, 0] = min(255, max(0, wave_x))  # Blue
                frame[y, x, 1] = min(255, max(0, int(128 + 80 * np.sin(x * 0.005 + y * 0.005 + t))))  # Green
                frame[y, x, 2] = min(255, max(0, wave_y))  # Red
        
        # Add some geometric shapes for object detection testing
        self._add_test_objects(frame, t)
        
        # Add text overlay
        self._add_text_overlay(frame)
        
        return frame
    
    def _add_test_objects(self, frame: np.ndarray, t: float) -> None:
        """Add geometric shapes that move around for testing object detection."""
        # Moving rectangle (simulates a "car")
        rect_x = int(200 + 100 * np.sin(t * 0.5))
        rect_y = int(200 + 50 * np.cos(t * 0.3))
        rect_w, rect_h = 120, 80
        
        # Ensure rectangle stays within bounds
        rect_x = max(0, min(self.width - rect_w, rect_x))
        rect_y = max(0, min(self.height - rect_h, rect_y))
        
        frame[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = [0, 255, 255]  # Yellow rectangle
        
        # Moving circle (simulates a "person")
        circle_x = int(self.width // 2 + 200 * np.cos(t * 0.8))
        circle_y = int(self.height // 2 + 150 * np.sin(t * 0.6))
        circle_radius = 40
        
        # Simple circle drawing
        for dy in range(-circle_radius, circle_radius + 1):
            for dx in range(-circle_radius, circle_radius + 1):
                if dx*dx + dy*dy <= circle_radius*circle_radius:
                    x, y = circle_x + dx, circle_y + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        frame[y, x] = [255, 0, 0]  # Blue circle
    
    def _add_text_overlay(self, frame: np.ndarray) -> None:
        """Add text overlay to the frame."""
        # Simple text rendering (mock camera identifier)
        text_y = 50
        text_color = [255, 255, 255]  # White
        
        # Add a simple text-like pattern (since we don't have cv2.putText)
        # This creates a basic "MOCK CAM" pattern
        for i, char_pattern in enumerate([
            # M
            [[1,0,1], [1,1,1], [1,0,1], [1,0,1], [1,0,1]],
            # O  
            [[1,1,1], [1,0,1], [1,0,1], [1,0,1], [1,1,1]],
            # C
            [[1,1,1], [1,0,0], [1,0,0], [1,0,0], [1,1,1]],
            # K
            [[1,0,1], [1,1,0], [1,1,0], [1,0,1], [1,0,1]]
        ]):
            for row, pattern_row in enumerate(char_pattern):
                for col, pixel in enumerate(pattern_row):
                    if pixel:
                        x = 50 + i * 30 + col * 4
                        y = text_y + row * 4
                        if 0 <= x < self.width - 4 and 0 <= y < self.height - 4:
                            frame[y:y+4, x:x+4] = text_color
    
    async def release(self) -> None:
        """Release mock camera resources."""
        await self.stop_capture()
        print("🔄 Mock camera resources released")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get mock camera capabilities."""
        return {
            "resolutions": ["1920x1080", "1280x720", "640x480"],
            "fps_ranges": [30, 60],
            "formats": ["BGR", "RGB"],
            "parameters": ["brightness", "contrast", "saturation"],
            "mock": True
        }
    
    def set_parameter(self, parameter: str, value: Any) -> bool:
        """Set a mock camera parameter."""
        try:
            if parameter == "brightness":
                self.brightness = max(0, min(100, int(value)))
                return True
            elif parameter == "contrast":
                self.contrast = max(0, min(100, int(value)))
                return True
            elif parameter == "saturation":
                self.saturation = max(0, min(100, int(value)))
                return True
            else:
                return False
        except (ValueError, TypeError):
            return False
    
    def get_parameter(self, parameter: str) -> Any:
        """Get a mock camera parameter value."""
        if parameter == "brightness":
            return self.brightness
        elif parameter == "contrast":
            return self.contrast
        elif parameter == "saturation":
            return self.saturation
        else:
            return None


# Register the mock camera with the factory
CameraFactory.register_camera("mock", MockCamera)

# Also register as fallback for other types during testing
CameraFactory.register_camera("jetson_csi", MockCamera)
CameraFactory.register_camera("mac_studio", MockCamera)

print("🎭 Mock camera implementation registered")