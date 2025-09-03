#!/usr/bin/env python3
"""
Native YOLO11 service for macOS with Apple Silicon GPU support.
Runs outside of containers to access Metal Performance Shaders.
"""

import asyncio
import os
import sys
import time
import logging
import pickle
import json
from pathlib import Path
from typing import Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import redis
import torch
from ultralytics import YOLO
from pydantic import Field
from pydantic_settings import BaseSettings

# Import shared models from the containerized services
from services.shared.models import (
    FrameMetadata, BoundingBox, DetectionResult, 
    DetectionMessage, FrameMessage
)


class NativeYOLOConfig(BaseSettings):
    """Configuration for native YOLO service."""
    
    # Redis connection
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # YOLO configuration
    yolo_model: str = Field(default="yolo11n.pt", env="YOLO_MODEL")
    yolo_confidence: float = Field(default=0.5, env="YOLO_CONFIDENCE")
    yolo_device: str = Field(default="mps", env="YOLO_DEVICE")  # Apple Silicon
    yolo_frame_stride: int = Field(default=1, env="YOLO_FRAME_STRIDE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class NativeYOLOService:
    """Native YOLO service with Apple Silicon GPU support."""
    
    def __init__(self):
        self.config = NativeYOLOConfig()
        self.setup_logging()
        
        self.model: Optional[YOLO] = None
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        
        # Performance tracking
        self.frames_processed = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()
        
    def setup_logging(self):
        """Setup structured logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "native.yolo", "message": "%(message)s", "component": "native_yolo"}',
            datefmt='%Y-%m-%dT%H:%M:%S.%fZ'
        )
        self.logger = logging.getLogger("native.yolo")
        
    async def load_model(self):
        """Load YOLO model with Apple Silicon optimization."""
        try:
            self.logger.info(f"Loading YOLO model: {self.config.yolo_model}")
            
            # Check for Apple Silicon MPS support
            if self.config.yolo_device == "mps" and torch.backends.mps.is_available():
                self.logger.info("Apple Silicon MPS detected and available")
                device = "mps"
            elif self.config.yolo_device == "cuda" and torch.cuda.is_available():
                device = "cuda"
                self.logger.info("CUDA GPU detected and available")
            else:
                device = "cpu"
                self.logger.info("Using CPU for inference")
            
            # Load model
            self.model = YOLO(self.config.yolo_model)
            
            # Move to appropriate device
            if device != "cpu":
                self.model.to(device)
                self.logger.info(f"Model loaded on {device}")
            else:
                self.logger.info("Model loaded on CPU")
                
            # Warm up model
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            self.model(dummy_image, verbose=False)
            self.logger.info("YOLO model warmed up successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
            
    def connect_redis(self):
        """Connect to Redis message bus."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                decode_responses=False  # Need binary data for frames
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")
            
            # Setup pub/sub
            self.pubsub = self.redis_client.pubsub()
            self.pubsub.subscribe("frame:camera.frames")
            self.logger.info("Subscribed to frame:camera.frames channel")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    def process_frame_array(self, frame: np.ndarray, metadata_dict: dict) -> Optional[DetectionMessage]:
        """Process a single frame array with YOLO detection."""
        try:
            start_time = time.perf_counter()
            
            # Create frame metadata
            frame_metadata = FrameMetadata(**metadata_dict)
            
            # Skip frames based on stride
            self.frames_processed += 1
            if self.frames_processed % self.config.yolo_frame_stride != 0:
                return None
            
            # Run YOLO inference on the actual frame
            results = self.model(frame, conf=self.config.yolo_confidence, verbose=False)
            
            # Convert results to our format
            detections = []
            for result in results:
                if result.boxes is not None:
                    self.logger.info(f"Found {len(result.boxes)} objects in frame {frame_metadata.frame_id}")
                    for box in result.boxes:
                        # Extract box coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        bbox = BoundingBox(
                            x1=float(x1), y1=float(y1),
                            x2=float(x2), y2=float(y2),
                            confidence=confidence,
                            class_name=class_name,
                            class_id=class_id
                        )
                        detections.append(bbox)
            
            # Create detection message
            processing_time = (time.perf_counter() - start_time) * 1000
            self.total_processing_time += processing_time
            self.frames_processed += 1
            
            # Create DetectionResult with all bounding boxes
            if detections:
                self.logger.info(f"Found {len(detections)} objects in frame {frame_metadata.frame_id}")
            
            detection_result = DetectionResult(
                bounding_boxes=detections,  # detections is already a list of BoundingBox
                processing_time_ms=processing_time,
                model_name="yolo11n"
            )
            
            detection_message = DetectionMessage(
                frame_id=frame_metadata.frame_id,
                result=detection_result,
                source_service="yolo_native"
            )
            
            # Log performance periodically
            if self.frames_processed % 10 == 0:
                avg_time = self.total_processing_time / self.frames_processed
                fps = 1000 / avg_time if avg_time > 0 else 0
                self.logger.info(
                    f"Processed {self.frames_processed} frames, "
                    f"avg: {avg_time:.1f}ms, fps: {fps:.1f}"
                )
                
            return detection_message
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return None
            
    def publish_detection(self, detection_message: DetectionMessage):
        """Publish detection results to Redis."""
        try:
            import json
            message_json = json.dumps(detection_message.model_dump(mode='json'))
            self.redis_client.publish("msg:detection.yolo", message_json.encode('utf-8'))
            self.logger.info(f"Published detection for frame {detection_message.frame_id} to msg:detection.yolo")
        except Exception as e:
            self.logger.error(f"Failed to publish detection: {e}")
            
    async def run(self):
        """Main service loop."""
        self.logger.info("Starting Native YOLO service")
        
        try:
            # Initialize components
            await self.load_model()
            self.connect_redis()
            
            self.logger.info("Native YOLO service started successfully")
            
            # Run the blocking Redis listener in a thread
            import asyncio
            import threading
            
            def process_messages():
                """Process Redis messages in a thread."""
                self.logger.info("Message processing thread started")
                for message in self.pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            self.logger.info(f"Processing frame from Redis")
                            # Unpickle frame package
                            import pickle
                            frame_package = pickle.loads(message['data'])
                            
                            # Extract frame data and metadata
                            frame_bytes = frame_package['frame_data']
                            metadata_dict = frame_package['metadata']
                            shape = frame_package['shape']
                            dtype = frame_package['dtype']
                            
                            # Reconstruct numpy array
                            frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)
                            
                            # Process frame with YOLO
                            detection_result = self.process_frame_array(frame, metadata_dict)
                            
                            # Publish results
                            if detection_result:
                                self.publish_detection(detection_result)
                                
                        except Exception as e:
                            self.logger.error(f"Error processing message: {e}")
            
            # Start processing thread
            thread = threading.Thread(target=process_messages, daemon=True)
            thread.start()
            
            # Keep the async function running
            while True:
                await asyncio.sleep(1)
                        
        except KeyboardInterrupt:
            self.logger.info("Shutting down Native YOLO service")
        except Exception as e:
            self.logger.error(f"Fatal error in Native YOLO service: {e}")
            raise
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """Cleanup resources."""
        if self.pubsub:
            self.pubsub.close()
        if self.redis_client:
            self.redis_client.close()
        self.logger.info("Native YOLO service stopped")


async def main():
    """Main entry point."""
    service = NativeYOLOService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
