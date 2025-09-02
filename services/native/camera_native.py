#!/usr/bin/env python3
"""
Native Camera service for macOS with direct hardware access.
Runs outside of containers to access Apple Studio Display camera.
"""

import asyncio
import os
import sys
import time
import logging
import base64
import pickle
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import redis
from pydantic import Field
from pydantic_settings import BaseSettings

# Import shared models from the containerized services
from services.shared.models import FrameMetadata, FrameMessage


class NativeCameraConfig(BaseSettings):
    """Configuration for native camera service."""
    
    # Redis connection
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Camera configuration
    camera_index: int = Field(default=0, env="CAMERA_INDEX")
    camera_width: int = Field(default=1280, env="CAMERA_WIDTH")
    camera_height: int = Field(default=720, env="CAMERA_HEIGHT")
    camera_fps: int = Field(default=30, env="CAMERA_FPS")
    
    # Performance
    frame_skip: int = Field(default=1, env="CAMERA_FRAME_SKIP")  # Process every Nth frame
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class NativeCameraService:
    """Native camera service with direct hardware access."""
    
    def __init__(self):
        self.config = NativeCameraConfig()
        self.setup_logging()
        
        self.camera: Optional[cv2.VideoCapture] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Performance tracking
        self.frames_captured = 0
        self.frames_published = 0
        self.start_time = time.time()
        
    def setup_logging(self):
        """Setup structured logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "native.camera", "message": "%(message)s", "component": "native_camera"}',
            datefmt='%Y-%m-%dT%H:%M:%S.%fZ'
        )
        self.logger = logging.getLogger("native.camera")
        
    def connect_redis(self):
        """Connect to Redis message bus."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                decode_responses=False  # We need binary data for pickled frames
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    def initialize_camera(self):
        """Initialize camera with optimal settings for Apple Studio Display."""
        try:
            self.logger.info(f"Initializing camera at index {self.config.camera_index}")
            
            # Use AVFoundation backend on macOS for better performance
            self.camera = cv2.VideoCapture(self.config.camera_index, cv2.CAP_AVFOUNDATION)
            
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera at index {self.config.camera_index}")
                
            # Configure camera settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            
            # Get actual settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(
                f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}fps"
            )
            
            # Warm up camera
            for _ in range(5):
                ret, frame = self.camera.read()
                if ret:
                    break
                time.sleep(0.1)
                
            if not ret:
                raise RuntimeError("Failed to capture initial frame")
                
            self.logger.info("Camera warmed up successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            raise
            
    def capture_and_publish_frame(self):
        """Capture a frame and publish it to Redis."""
        try:
            ret, frame = self.camera.read()
            if not ret:
                self.logger.warning("Failed to capture frame")
                return False
                
            self.frames_captured += 1
            
            # Skip frames based on frame_skip setting
            if self.frames_captured % self.config.frame_skip != 0:
                return True
                
            # Create frame metadata
            frame_id = int(time.time() * 1000000) + self.frames_captured  # Use microseconds + counter for unique int
            timestamp = datetime.utcnow()
            
            frame_metadata = FrameMetadata(
                frame_id=frame_id,
                timestamp=timestamp,
                width=frame.shape[1],
                height=frame.shape[0],
                channels=frame.shape[2] if len(frame.shape) > 2 else 1,
                fps=self.config.camera_fps,
                camera_id=f"camera_{self.config.camera_index}"
            )
            
            # Encode frame as JPEG for efficient transmission
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if not ret:
                self.logger.warning("Failed to encode frame")
                return False
                
            # Create frame package in the format expected by containerized services
            frame_package = {
                'metadata': frame_metadata.model_dump(),
                'frame_data': frame.tobytes(),
                'shape': frame.shape,
                'dtype': str(frame.dtype),
                'compressed': False
            }
            
            # Serialize package
            package_data = pickle.dumps(frame_package)
            
            # Publish to Redis (use same format as containerized services)
            self.redis_client.publish("frame:camera.frames", package_data)
            
            self.frames_published += 1
            
            # Log performance periodically
            if self.frames_published % 30 == 0:  # Every 30 published frames
                elapsed = time.time() - self.start_time
                capture_fps = self.frames_captured / elapsed if elapsed > 0 else 0
                publish_fps = self.frames_published / elapsed if elapsed > 0 else 0
                
                self.logger.info(
                    f"Performance: Captured {self.frames_captured} frames ({capture_fps:.1f} fps), "
                    f"Published {self.frames_published} frames ({publish_fps:.1f} fps)"
                )
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error capturing/publishing frame: {e}")
            return False
            
    async def run(self):
        """Main service loop."""
        self.logger.info("Starting Native Camera service")
        
        try:
            # Initialize components
            self.connect_redis()
            self.initialize_camera()
            
            self.logger.info("Native Camera service started successfully")
            self.logger.info(f"Publishing frames to 'frame:camera.frames' channel")
            
            # Main capture loop
            while True:
                success = self.capture_and_publish_frame()
                if not success:
                    self.logger.warning("Frame capture failed, retrying...")
                    await asyncio.sleep(0.1)
                else:
                    # Small delay to control frame rate
                    await asyncio.sleep(1.0 / self.config.camera_fps)
                    
        except KeyboardInterrupt:
            self.logger.info("Shutting down Native Camera service")
        except Exception as e:
            self.logger.error(f"Fatal error in Native Camera service: {e}")
            raise
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """Cleanup resources."""
        if self.camera:
            self.camera.release()
            self.logger.info("Camera released")
            
        if self.redis_client:
            self.redis_client.close()
            
        self.logger.info("Native Camera service stopped")


async def main():
    """Main entry point."""
    service = NativeCameraService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
