"""Mock camera implementation for testing the vision pipeline."""

import asyncio
import time
import numpy as np
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

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
                timestamp=datetime.utcnow(),
                camera_id=self.camera_id,
                width=self.width,
                height=self.height,
                channels=3,  # BGR has 3 channels
                fps=self.fps
            )
            
            return frame, metadata
            
        except Exception as e:
            print(f"❌ Mock camera frame generation failed: {e}")
            return None
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate a synthetic frame with animated patterns using efficient NumPy operations."""
        # Add time-based animation
        t = time.time() * 2  # Animation speed
        
        # Create coordinate grids using NumPy meshgrid (vectorized)
        x = np.arange(self.width, dtype=np.float32)
        y = np.arange(self.height, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        
        # Create moving wave patterns using vectorized NumPy operations
        wave_x = 128 + 100 * np.sin(X * 0.01 + t)
        wave_y = 128 + 100 * np.cos(Y * 0.01 + t * 0.7)
        green_pattern = 128 + 80 * np.sin(X * 0.005 + Y * 0.005 + t)
        
        # Clip values to valid range and convert to uint8
        wave_x = np.clip(wave_x, 0, 255).astype(np.uint8)
        wave_y = np.clip(wave_y, 0, 255).astype(np.uint8)
        green_pattern = np.clip(green_pattern, 0, 255).astype(np.uint8)
        
        # Create frame with RGB channels
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = wave_x  # Blue channel
        frame[:, :, 1] = green_pattern  # Green channel
        frame[:, :, 2] = wave_y  # Red channel
        
        # Add some geometric shapes for object detection testing
        self._add_test_objects(frame, t)
        
        # Add text overlay
        self._add_text_overlay(frame)
        
        return frame
    
    def _add_test_objects(self, frame: np.ndarray, t: float) -> None:
        """Add realistic objects that YOLO can detect."""
        # Car-like object (larger, more distinctive)
        car_x = int(300 + 200 * np.sin(t * 0.3))
        car_y = int(400 + 80 * np.cos(t * 0.2))
        car_w, car_h = 220, 120  # Larger size
        
        # Ensure car stays within bounds
        car_x = max(0, min(self.width - car_w, car_x))
        car_y = max(0, min(self.height - car_h, car_y))
        
        # Car body (dark blue/gray)
        frame[car_y:car_y+car_h, car_x:car_x+car_w] = [80, 80, 120]
        # Car roof (lighter)
        roof_h = car_h // 3
        frame[car_y:car_y+roof_h, car_x+20:car_x+car_w-20] = [120, 120, 160]
        # Car windows (dark)
        frame[car_y+5:car_y+roof_h-5, car_x+30:car_x+car_w-30] = [20, 20, 40]
        
        # Person-like object (larger, more distinctive)
        person_x = int(self.width // 2 + 300 * np.cos(t * 0.6))
        person_y = int(250 + 120 * np.sin(t * 0.4))
        person_w, person_h = 80, 180  # Larger person
        
        # Ensure person stays within bounds  
        person_x = max(0, min(self.width - person_w, person_x))
        person_y = max(0, min(self.height - person_h, person_y))
        
        # Person body (skin tone)
        frame[person_y+30:person_y+person_h, person_x:person_x+person_w] = [180, 140, 120]
        # Person head (round-ish)
        head_x = person_x + person_w // 2
        head_y = person_y + 15
        head_radius = 25
        
        # Efficient circle drawing using NumPy for head
        y_min = max(0, head_y - head_radius)
        y_max = min(self.height, head_y + head_radius + 1)
        x_min = max(0, head_x - head_radius)
        x_max = min(self.width, head_x + head_radius + 1)
        
        if y_min < y_max and x_min < x_max:
            # Create coordinate grids for the head region
            yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
            # Calculate distance from head center
            head_mask = (xx - head_x)**2 + (yy - head_y)**2 <= head_radius**2
            # Apply skin tone to head
            frame[y_min:y_max, x_min:x_max][head_mask] = [200, 160, 140]  # Head color
            
        # Add a large bottle/cup object (common YOLO class)
        bottle_x = int(150 + 100 * np.sin(t * 0.9))
        bottle_y = int(500 + 50 * np.cos(t * 0.7))
        bottle_w, bottle_h = 60, 140  # Much larger bottle
        
        # Ensure bottle stays within bounds
        bottle_x = max(0, min(self.width - bottle_w, bottle_x))
        bottle_y = max(0, min(self.height - bottle_h, bottle_y))
        
        # Bottle body (green/brown)
        frame[bottle_y:bottle_y+bottle_h, bottle_x:bottle_x+bottle_w] = [60, 120, 60]
        # Bottle cap (darker)
        cap_h = bottle_h // 6
        frame[bottle_y:bottle_y+cap_h, bottle_x:bottle_x+bottle_w] = [40, 80, 40]
    
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