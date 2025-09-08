"""Simple mock camera that uses a static test image."""

import asyncio
import time
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from .base import CameraInterface, CameraFactory
from ..shared.models import FrameMetadata

# Register this camera type when the module is imported
def register():
    """Register the simple mock camera with the factory."""
    CameraFactory.register_camera("simple_mock", SimpleMockCamera)


class SimpleMockCamera(CameraInterface):
    """Simple mock camera that uses a static test image for YOLO testing."""
    
    def __init__(self, camera_id: str = "simple_mock_camera"):
        super().__init__(camera_id)
        self.width = 1920
        self.height = 1080
        self.fps = 10  # Lower FPS for testing
        self.frame_interval = 1.0 / self.fps
        self.last_frame_time = 0
        self.test_frame = None
        
    async def initialize(self, **kwargs) -> bool:
        """Initialize the mock camera with a test image."""
        try:
            self.width = kwargs.get('width', 1920)
            self.height = kwargs.get('height', 1080)
            self.fps = kwargs.get('fps', 10)
            self.frame_interval = 1.0 / self.fps
            
            # Create or load a test image
            self.test_frame = self._create_test_frame()
            
            print(f"🎥 Simple mock camera initialized with static test image: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"❌ Simple mock camera initialization failed: {e}")
            return False
    
    def _create_test_frame(self) -> np.ndarray:
        """Create a simple test frame with objects YOLO might detect."""
        # Create a frame with a simple scene
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Add a large dark rectangle that might be detected as a TV/monitor
        cv2.rectangle(frame, (400, 200), (1520, 880), (30, 30, 30), -1)
        cv2.rectangle(frame, (420, 220), (1500, 860), (100, 100, 100), -1)
        
        # Add text using OpenCV if available
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "YOLO TEST", (700, 500), font, 3, (255, 255, 255), 5)
            cv2.putText(frame, "Static Test Image", (650, 600), font, 1.5, (200, 200, 200), 3)
            
            # Add some circles that might be detected as sports balls
            cv2.circle(frame, (300, 540), 100, (255, 100, 0), -1)  # Orange circle (basketball?)
            cv2.circle(frame, (1620, 540), 80, (0, 255, 0), -1)  # Green circle (tennis ball?)
            
            # Add a rectangle that might be detected as a laptop/book
            cv2.rectangle(frame, (100, 800), (400, 950), (150, 100, 50), -1)
            
        except:
            # If cv2 isn't available, just use the basic frame
            pass
            
        return frame
    
    async def start_capture(self) -> None:
        """Start mock camera capture."""
        self.is_active = True
        self.reset_frame_count()
        self.last_frame_time = time.time()
        print("📹 Simple mock camera capture started with static image")
    
    async def stop_capture(self) -> None:
        """Stop mock camera capture."""
        self.is_active = False
        print("🛑 Simple mock camera capture stopped")
    
    async def capture_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Capture a frame (returns the same static test image)."""
        if not self.is_active or self.test_frame is None:
            return None
            
        current_time = time.time()
        
        # Throttle frame rate
        if current_time - self.last_frame_time < self.frame_interval:
            await asyncio.sleep(self.frame_interval - (current_time - self.last_frame_time))
            current_time = time.time()
            
        self.last_frame_time = current_time
        
        # Create metadata
        metadata = FrameMetadata(
            frame_id=self.get_frame_count(),
            timestamp=datetime.utcnow(),
            width=self.width,
            height=self.height,
            channels=3,
            fps=self.fps,
            camera_id=self.camera_id,
            processing_stage="capture"
        )
        
        self.increment_frame_count()
        
        # Return a copy of the test frame (to simulate real capture)
        return self.test_frame.copy(), metadata
    
    async def get_properties(self) -> Dict[str, Any]:
        """Get camera properties."""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "type": "simple_mock",
            "static_image": True
        }
    
    async def set_property(self, property_name: str, value: Any) -> bool:
        """Set a camera property (no-op for mock)."""
        print(f"Mock camera: set_property({property_name}={value}) - ignored")
        return True
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.stop_capture()
        print("🧹 Simple mock camera cleaned up")


# Register the camera when module is imported
register()
