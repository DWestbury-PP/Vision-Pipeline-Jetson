"""Abstract base class for camera implementations."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np
import asyncio
from ..shared.models import FrameMetadata


class CameraInterface(ABC):
    """Abstract base class for camera implementations."""
    
    def __init__(self, camera_id: str = "default"):
        self.camera_id = camera_id
        self.is_active = False
        self._frame_count = 0
        
    @abstractmethod
    async def initialize(self, **kwargs) -> bool:
        """Initialize the camera.
        
        Args:
            **kwargs: Camera-specific initialization parameters
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def start_capture(self) -> None:
        """Start camera capture."""
        pass
    
    @abstractmethod
    async def stop_capture(self) -> None:
        """Stop camera capture."""
        pass
    
    @abstractmethod
    async def capture_frame(self) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Capture a single frame.
        
        Returns:
            Tuple of (frame_data, metadata) or None if capture failed
        """
        pass
    
    @abstractmethod
    async def release(self) -> None:
        """Release camera resources."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get camera capabilities.
        
        Returns:
            Dictionary of camera capabilities
        """
        pass
    
    @abstractmethod
    def set_parameter(self, parameter: str, value: Any) -> bool:
        """Set a camera parameter.
        
        Args:
            parameter: Parameter name
            value: Parameter value
            
        Returns:
            True if parameter was set successfully
        """
        pass
    
    @abstractmethod
    def get_parameter(self, parameter: str) -> Any:
        """Get a camera parameter value.
        
        Args:
            parameter: Parameter name
            
        Returns:
            Parameter value or None if not supported
        """
        pass
    
    def get_next_frame_id(self) -> int:
        """Get the next frame ID."""
        self._frame_count += 1
        return self._frame_count
    
    def reset_frame_count(self) -> None:
        """Reset the frame counter."""
        self._frame_count = 0


class StreamingCameraInterface(CameraInterface):
    """Extended interface for streaming cameras with async frame generation."""
    
    def __init__(self, camera_id: str = "default"):
        super().__init__(camera_id)
        self._capture_task: Optional[asyncio.Task] = None
        self._frame_queue: Optional[asyncio.Queue] = None
        self._stop_event: Optional[asyncio.Event] = None
    
    async def start_streaming(self, frame_callback=None, max_queue_size: int = 10) -> None:
        """Start continuous frame streaming.
        
        Args:
            frame_callback: Optional callback for each frame
            max_queue_size: Maximum size of internal frame queue
        """
        if self._capture_task and not self._capture_task.done():
            return  # Already streaming
            
        self._frame_queue = asyncio.Queue(maxsize=max_queue_size)
        self._stop_event = asyncio.Event()
        
        self._capture_task = asyncio.create_task(
            self._streaming_loop(frame_callback)
        )
    
    async def stop_streaming(self) -> None:
        """Stop continuous frame streaming."""
        if self._stop_event:
            self._stop_event.set()
            
        if self._capture_task:
            try:
                await asyncio.wait_for(self._capture_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._capture_task.cancel()
                try:
                    await self._capture_task
                except asyncio.CancelledError:
                    pass
    
    async def get_latest_frame(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Get the latest frame from the streaming queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Latest frame and metadata or None if timeout
        """
        if not self._frame_queue:
            return None
            
        try:
            return await asyncio.wait_for(self._frame_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    async def _streaming_loop(self, frame_callback=None) -> None:
        """Internal streaming loop."""
        while not self._stop_event.is_set():
            try:
                frame_data = await self.capture_frame()
                if frame_data:
                    # Add to queue (drop oldest if full)
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    
                    await self._frame_queue.put(frame_data)
                    
                    # Call callback if provided
                    if frame_callback:
                        try:
                            await frame_callback(*frame_data)
                        except Exception:
                            # Don't let callback errors stop streaming
                            pass
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)
                
            except Exception:
                # Log error but continue streaming
                await asyncio.sleep(0.1)
    
    async def release(self) -> None:
        """Release camera resources and stop streaming."""
        await self.stop_streaming()
        await super().release()


class CameraFactory:
    """Factory for creating camera instances based on configuration."""
    
    _camera_classes = {}
    
    @classmethod
    def register_camera(cls, camera_type: str, camera_class: type):
        """Register a camera implementation.
        
        Args:
            camera_type: Type identifier (e.g., "mac_studio", "jetson")
            camera_class: Camera class to register
        """
        cls._camera_classes[camera_type] = camera_class
    
    @classmethod
    def create_camera(cls, camera_type: str, camera_id: str = "default") -> CameraInterface:
        """Create a camera instance.
        
        Args:
            camera_type: Type of camera to create
            camera_id: Camera identifier
            
        Returns:
            Camera instance
            
        Raises:
            ValueError: If camera type is not supported
        """
        if camera_type not in cls._camera_classes:
            raise ValueError(f"Unsupported camera type: {camera_type}")
            
        return cls._camera_classes[camera_type](camera_id)
    
    @classmethod
    def get_supported_cameras(cls) -> list:
        """Get list of supported camera types."""
        return list(cls._camera_classes.keys())
