"""YOLO11 object detection service."""

import asyncio
import time
import numpy as np
from typing import Optional, List
from ultralytics import YOLO
import torch

from ..message_bus.redis_bus import RedisMessageBus
from ..message_bus.base import MessageBusSubscriber, MessageBusPublisher, Channels
from ..shared.models import (
    FrameMetadata, BoundingBox, DetectionResult, 
    DetectionMessage, FrameMessage
)
from ..shared.logging_config import setup_logging, log_performance_metrics, log_error_with_context
from ..shared.config import get_config


class YOLOService:
    """YOLO11 object detection service."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("yolo_service")
        
        # YOLO model
        self.model: Optional[YOLO] = None
        self.device = self.config.yolo.device
        
        # Message bus
        self.message_bus = RedisMessageBus()
        self.subscriber = MessageBusSubscriber(self.message_bus)
        self.publisher = MessageBusPublisher(self.message_bus)
        
        # Service state
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.frames_processed = 0
        self.total_processing_time = 0.0
        self.last_frame_time = 0
        
        # Frame processing control
        self.frame_stride = self.config.yolo.frame_stride
        self.frame_counter = 0
        
    async def start(self) -> None:
        """Start the YOLO service."""
        try:
            self.logger.info("Starting YOLO service")
            
            # Load YOLO model
            await self._load_model()
            
            # Initialize message bus
            await self.subscriber.start()
            await self.publisher.start()
            
            # Subscribe to camera frames
            await self.subscriber.subscribe_to_frames(
                Channels.CAMERA_FRAMES,
                self._process_frame
            )
            
            self.is_running = True
            
            self.logger.info(
                "YOLO service started successfully",
                extra={
                    "model": self.config.yolo.model,
                    "device": self.device,
                    "confidence": self.config.yolo.confidence,
                    "frame_stride": self.frame_stride
                }
            )
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="yolo_service_start")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the YOLO service."""
        self.logger.info("Stopping YOLO service")
        self.is_running = False
        
        # Stop message bus
        try:
            await self.subscriber.stop()
            await self.publisher.stop()
        except Exception as e:
            log_error_with_context(self.logger, e, operation="message_bus_stop")
        
        # Clear model from memory
        if self.model:
            del self.model
            self.model = None
            
        # Clear GPU cache if using CUDA/MPS
        if self.device in ["cuda", "mps"]:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            elif hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        
        self.logger.info(f"YOLO service stopped. Processed {self.frames_processed} frames")
    
    async def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            self.logger.info(f"Loading YOLO model: {self.config.yolo.model}")
            
            # Load model in thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: YOLO(self.config.yolo.model)
            )
            
            # Move model to device
            if self.device == "mps" and torch.backends.mps.is_available():
                self.model.to("mps")
                self.logger.info("Model loaded on Apple Silicon MPS")
            elif self.device == "cuda" and torch.cuda.is_available():
                self.model.to("cuda")
                self.logger.info("Model loaded on CUDA")
            else:
                self.device = "cpu"
                self.logger.info("Model loaded on CPU")
            
            # Warm up the model with a dummy frame
            dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            await loop.run_in_executor(
                None,
                lambda: self.model.predict(dummy_frame, verbose=False)
            )
            
            self.logger.info("YOLO model warmed up successfully")
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"model_path": self.config.yolo.model, "device": self.device},
                "model_loading"
            )
            raise
    
    async def _process_frame(self, frame: np.ndarray, metadata: FrameMetadata) -> None:
        """Process a frame with YOLO detection."""
        try:
            self.frame_counter += 1
            
            # Skip frames based on stride
            if self.frame_counter % self.frame_stride != 0:
                return
            
            start_time = time.perf_counter()
            
            # Run YOLO inference in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._run_inference,
                frame
            )
            
            if results:
                # Parse results
                bounding_boxes = self._parse_yolo_results(results[0])
                
                processing_time = (time.perf_counter() - start_time) * 1000
                self.total_processing_time += processing_time
                self.frames_processed += 1
                
                # Create detection result
                detection_result = DetectionResult(
                    bounding_boxes=bounding_boxes,
                    processing_time_ms=processing_time,
                    model_name=f"yolo11_{self.config.yolo.model}",
                    timestamp=metadata.timestamp
                )
                
                # Publish result
                detection_message = DetectionMessage(
                    source_service="yolo_service",
                    result=detection_result,
                    frame_id=metadata.frame_id
                )
                
                await self.publisher.publish_message(
                    Channels.YOLO_RESULTS,
                    detection_message
                )
                
                # Log performance metrics
                if (self.config.pipeline.enable_performance_metrics and 
                    self.frames_processed % 10 == 0):
                    avg_processing_time = self.total_processing_time / self.frames_processed
                    log_performance_metrics(
                        self.logger,
                        "yolo_inference",
                        processing_time,
                        frame_id=metadata.frame_id,
                        model_name=f"yolo11_{self.config.yolo.model}",
                        detections_count=len(bounding_boxes),
                        avg_processing_time=avg_processing_time
                    )
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"frame_id": metadata.frame_id},
                "frame_processing"
            )
    
    def _run_inference(self, frame: np.ndarray):
        """Run YOLO inference (called in thread pool)."""
        try:
            results = self.model.predict(
                frame,
                conf=self.config.yolo.confidence,
                verbose=False,
                device=self.device
            )
            return results
        except Exception as e:
            self.logger.error(f"YOLO inference failed: {e}")
            return None
    
    def _parse_yolo_results(self, result) -> List[BoundingBox]:
        """Parse YOLO results into BoundingBox objects."""
        bounding_boxes = []
        
        try:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    confidence = float(confidences[i])
                    class_id = int(class_ids[i])
                    
                    # Get class name from model
                    class_name = self.model.names.get(class_id, f"class_{class_id}")
                    
                    bbox = BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=confidence,
                        class_name=class_name,
                        class_id=class_id
                    )
                    
                    bounding_boxes.append(bbox)
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"result_type": type(result)},
                "parse_yolo_results"
            )
        
        return bounding_boxes
    
    async def get_service_info(self) -> dict:
        """Get service information and statistics."""
        try:
            avg_processing_time = (
                self.total_processing_time / self.frames_processed 
                if self.frames_processed > 0 else 0.0
            )
            
            return {
                "service_name": "yolo_service",
                "model": self.config.yolo.model,
                "device": self.device,
                "is_running": self.is_running,
                "frames_processed": self.frames_processed,
                "average_processing_time_ms": avg_processing_time,
                "confidence_threshold": self.config.yolo.confidence,
                "frame_stride": self.frame_stride,
                "model_classes": len(self.model.names) if self.model else 0
            }
        except Exception as e:
            log_error_with_context(self.logger, e, operation="get_service_info")
            return {}
    
    async def update_confidence(self, confidence: float) -> bool:
        """Update confidence threshold."""
        try:
            if 0.0 <= confidence <= 1.0:
                self.config.yolo.confidence = confidence
                self.logger.info(
                    f"Updated confidence threshold to {confidence}",
                    extra={"confidence": confidence}
                )
                return True
            return False
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"confidence": confidence},
                "update_confidence"
            )
            return False
    
    async def update_frame_stride(self, stride: int) -> bool:
        """Update frame processing stride."""
        try:
            if stride >= 1:
                self.frame_stride = stride
                self.logger.info(
                    f"Updated frame stride to {stride}",
                    extra={"frame_stride": stride}
                )
                return True
            return False
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"stride": stride},
                "update_frame_stride"
            )
            return False


async def main():
    """Main function for running the YOLO service standalone."""
    service = YOLOService()
    
    try:
        await service.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nShutting down YOLO service...")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
