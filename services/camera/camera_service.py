"""Camera service that captures frames and publishes them to the message bus."""

import asyncio
import time
from typing import Optional

from .base import CameraFactory
from . import jetson_csi  # Register Jetson CSI camera
from . import mock_camera  # Register Mock camera for testing
from . import simple_mock_camera  # Register Simple Mock camera with static image
from ..message_bus.redis_bus import RedisMessageBus
from ..message_bus.base import MessageBusPublisher, Channels
from ..shared.models import SystemStatus, StatusMessage
from ..shared.logging_config import setup_logging, log_performance_metrics, log_error_with_context
from ..shared.config import get_config


class CameraService:
    """Camera service that captures and publishes frames."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("camera_service")
        
        # Initialize camera with fallback to mock
        try:
            self.camera = CameraFactory.create_camera(
                self.config.camera_type,
                f"{self.config.camera_type}_{self.config.camera_index}"
            )
        except (ValueError, Exception):
            # Fallback to mock camera if real camera type fails
            self.logger.warning(f"Failed to create {self.config.camera_type} camera, falling back to mock camera")
            self.camera = CameraFactory.create_camera("mock", "mock_fallback")
        
        # Initialize message bus
        self.message_bus = RedisMessageBus()
        self.publisher = MessageBusPublisher(self.message_bus)
        
        # Service state
        self.is_running = False
        self.capture_task: Optional[asyncio.Task] = None
        self.status_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.frames_captured = 0
        self.start_time = 0
        self.last_status_time = 0
        self.status_interval = 5.0  # seconds
        
    async def start(self) -> None:
        """Start the camera service."""
        try:
            self.logger.info("Starting camera service")
            
            # Initialize message bus
            await self.publisher.start()
            
            # Initialize camera
            camera_init_params = {
                'width': self.config.camera_width,
                'height': self.config.camera_height,
                'fps': self.config.camera_fps,
                'index': self.config.camera_index
            }
            
            if not await self.camera.initialize(**camera_init_params):
                # Try fallback to mock camera if initialization fails
                if self.camera.camera_id != "mock_fallback":
                    self.logger.warning("Camera initialization failed, falling back to mock camera")
                    self.camera = CameraFactory.create_camera("mock", "mock_fallback")
                    if not await self.camera.initialize(**camera_init_params):
                        raise RuntimeError("Failed to initialize even mock camera")
                else:
                    raise RuntimeError("Failed to initialize camera")
            
            await self.camera.start_capture()
            
            self.is_running = True
            self.start_time = time.perf_counter()
            
            # Start capture and status tasks
            self.capture_task = asyncio.create_task(self._capture_loop())
            self.status_task = asyncio.create_task(self._status_loop())
            
            self.logger.info(
                "Camera service started successfully",
                extra={
                    "camera_type": self.config.camera_type,
                    "resolution": f"{self.config.camera_width}x{self.config.camera_height}",
                    "fps": self.config.camera_fps
                }
            )
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="camera_service_start")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the camera service."""
        self.logger.info("Stopping camera service")
        self.is_running = False
        
        # Cancel tasks
        if self.capture_task:
            self.capture_task.cancel()
            try:
                await self.capture_task
            except asyncio.CancelledError:
                pass
        
        if self.status_task:
            self.status_task.cancel()
            try:
                await self.status_task
            except asyncio.CancelledError:
                pass
        
        # Stop camera
        try:
            await self.camera.stop_capture()
            await self.camera.release()
        except Exception as e:
            log_error_with_context(self.logger, e, operation="camera_stop")
        
        # Stop message bus
        try:
            await self.publisher.stop()
        except Exception as e:
            log_error_with_context(self.logger, e, operation="message_bus_stop")
        
        self.logger.info("Camera service stopped")
    
    async def _capture_loop(self) -> None:
        """Main capture loop."""
        self.logger.info("Starting capture loop")
        
        while self.is_running:
            try:
                start_time = time.perf_counter()
                
                # Capture frame
                frame_data = await self.camera.capture_frame()
                if frame_data is None:
                    await asyncio.sleep(0.01)  # Brief pause before retry
                    continue
                
                frame, metadata = frame_data
                
                # Publish frame to message bus
                await self.publisher.publish_frame(
                    Channels.CAMERA_FRAMES,
                    frame,
                    metadata
                )
                
                self.frames_captured += 1
                
                # Log performance metrics periodically
                processing_time = (time.perf_counter() - start_time) * 1000
                if (self.config.enable_performance_metrics and 
                    self.frames_captured % 30 == 0):
                    log_performance_metrics(
                        self.logger,
                        "frame_publish",
                        processing_time,
                        frame_id=metadata.frame_id,
                        model_name="camera_service",
                        frame_size=frame.nbytes
                    )
                
                # Brief yield to allow other tasks to run
                await asyncio.sleep(0.001)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error_with_context(
                    self.logger,
                    e,
                    {"frames_captured": self.frames_captured},
                    "capture_loop"
                )
                # Brief pause before retry
                await asyncio.sleep(0.1)
        
        self.logger.info(f"Capture loop stopped. Frames captured: {self.frames_captured}")
    
    async def _status_loop(self) -> None:
        """Status reporting loop."""
        while self.is_running:
            try:
                current_time = time.perf_counter()
                
                if current_time - self.last_status_time >= self.status_interval:
                    await self._publish_status()
                    self.last_status_time = current_time
                
                await asyncio.sleep(1.0)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error_with_context(self.logger, e, operation="status_loop")
                await asyncio.sleep(1.0)
    
    async def _publish_status(self) -> None:
        """Publish system status."""
        try:
            current_time = time.perf_counter()
            elapsed_time = current_time - self.start_time
            average_fps = self.frames_captured / elapsed_time if elapsed_time > 0 else 0.0
            
            status = SystemStatus(
                camera_active=self.is_running,
                yolo_active=False,  # Will be updated by YOLO service
                moondream_instances=0,  # Will be updated by Moondream service
                message_bus_connected=await self.message_bus.health_check(),
                frames_processed=self.frames_captured,
                average_fps=average_fps,
                memory_usage_mb=0.0,  # TODO: Add memory monitoring
                gpu_utilization=None
            )
            
            status_message = StatusMessage(
                source_service="camera_service",
                status=status
            )
            
            await self.publisher.publish_message(Channels.SYSTEM_STATUS, status_message)
            
            self.logger.debug(
                f"Status published: {self.frames_captured} frames, {average_fps:.1f} FPS",
                extra={
                    "frames_captured": self.frames_captured,
                    "average_fps": average_fps,
                    "elapsed_time": elapsed_time
                }
            )
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="publish_status")
    
    async def get_camera_info(self) -> dict:
        """Get camera information and capabilities."""
        try:
            capabilities = self.camera.get_capabilities()
            return {
                "camera_id": self.camera.camera_id,
                "camera_type": self.config.camera_type,
                "is_active": self.camera.is_active,
                "frames_captured": self.frames_captured,
                "capabilities": capabilities
            }
        except Exception as e:
            log_error_with_context(self.logger, e, operation="get_camera_info")
            return {}
    
    async def set_camera_parameter(self, parameter: str, value) -> bool:
        """Set a camera parameter."""
        try:
            result = self.camera.set_parameter(parameter, value)
            self.logger.info(
                f"Camera parameter {parameter} set to {value}",
                extra={"parameter": parameter, "value": value, "success": result}
            )
            return result
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"parameter": parameter, "value": value},
                "set_camera_parameter"
            )
            return False
    
    async def get_camera_parameter(self, parameter: str):
        """Get a camera parameter value."""
        try:
            return self.camera.get_parameter(parameter)
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"parameter": parameter},
                "get_camera_parameter"
            )
            return None


async def main():
    """Main function for running the camera service standalone."""
    service = CameraService()
    
    try:
        await service.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nShutting down camera service...")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
