"""Fusion service that combines YOLO and VLM outputs into unified results."""

import asyncio
import time
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta

from ..message_bus.redis_bus import RedisMessageBus
from ..message_bus.base import MessageBusSubscriber, MessageBusPublisher, Channels
from ..shared.models import (
    DetectionResult, VLMResult, ProcessingPipelineResult, FrameMetadata,
    DetectionMessage, VLMMessage, BoundingBox, FusionResultMessage
)
from ..shared.logging_config import setup_logging, log_performance_metrics, log_error_with_context
from ..shared.config import get_config


class ResultCache:
    """Cache for storing and matching detection and VLM results."""
    
    def __init__(self, max_age_seconds: float = 5.0, max_size: int = 100):
        self.max_age = max_age_seconds
        self.max_size = max_size
        self.yolo_results: Dict[int, Tuple[DetectionResult, datetime]] = {}
        self.vlm_results: Dict[int, Tuple[VLMResult, datetime]] = {}
        self.frame_metadata: Dict[int, Tuple[FrameMetadata, datetime]] = {}
        
    def add_yolo_result(self, frame_id: int, result: DetectionResult) -> None:
        """Add a YOLO result to the cache."""
        self.yolo_results[frame_id] = (result, datetime.utcnow())
        self._cleanup_cache()
    
    def add_vlm_result(self, frame_id: int, result: VLMResult) -> None:
        """Add a VLM result to the cache."""
        self.vlm_results[frame_id] = (result, datetime.utcnow())
        self._cleanup_cache()
    
    def add_frame_metadata(self, frame_id: int, metadata: FrameMetadata) -> None:
        """Add frame metadata to the cache."""
        self.frame_metadata[frame_id] = (metadata, datetime.utcnow())
        self._cleanup_cache()
    
    def get_yolo_result(self, frame_id: int) -> Optional[DetectionResult]:
        """Get YOLO result for a frame."""
        if frame_id in self.yolo_results:
            result, timestamp = self.yolo_results[frame_id]
            if self._is_recent(timestamp):
                return result
        return None
    
    def get_vlm_result(self, frame_id: int) -> Optional[VLMResult]:
        """Get VLM result for a frame."""
        if frame_id in self.vlm_results:
            result, timestamp = self.vlm_results[frame_id]
            if self._is_recent(timestamp):
                return result
        return None
    
    def get_frame_metadata(self, frame_id: int) -> Optional[FrameMetadata]:
        """Get frame metadata."""
        if frame_id in self.frame_metadata:
            metadata, timestamp = self.frame_metadata[frame_id]
            if self._is_recent(timestamp):
                return metadata
        return None
    
    def get_latest_yolo_result(self) -> Optional[Tuple[int, DetectionResult]]:
        """Get the most recent YOLO result."""
        if not self.yolo_results:
            return None
            
        latest_frame_id = max(self.yolo_results.keys())
        result, timestamp = self.yolo_results[latest_frame_id]
        
        if self._is_recent(timestamp):
            return latest_frame_id, result
        return None
    
    def get_latest_vlm_result(self) -> Optional[Tuple[int, VLMResult]]:
        """Get the most recent VLM result."""
        if not self.vlm_results:
            return None
            
        latest_frame_id = max(self.vlm_results.keys())
        result, timestamp = self.vlm_results[latest_frame_id]
        
        if self._is_recent(timestamp):
            return latest_frame_id, result
        return None
    
    def _is_recent(self, timestamp: datetime) -> bool:
        """Check if a timestamp is within the max age."""
        return (datetime.utcnow() - timestamp).total_seconds() <= self.max_age
    
    def _cleanup_cache(self) -> None:
        """Remove old entries from the cache."""
        current_time = datetime.utcnow()
        
        # Remove old entries
        for cache in [self.yolo_results, self.vlm_results, self.frame_metadata]:
            expired_keys = []
            for frame_id, (_, timestamp) in cache.items():
                if (current_time - timestamp).total_seconds() > self.max_age:
                    expired_keys.append(frame_id)
            
            for key in expired_keys:
                cache.pop(key, None)
        
        # Limit cache size (keep most recent entries)
        for cache in [self.yolo_results, self.vlm_results, self.frame_metadata]:
            if len(cache) > self.max_size:
                # Sort by timestamp and keep the most recent
                sorted_items = sorted(cache.items(), key=lambda x: x[1][1], reverse=True)
                cache.clear()
                for frame_id, (result, timestamp) in sorted_items[:self.max_size]:
                    cache[frame_id] = (result, timestamp)


class FusionService:
    """Service that fuses YOLO and VLM results into unified outputs."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("fusion_service")
        
        # Result cache
        self.cache = ResultCache(max_age_seconds=10.0, max_size=50)
        
        # Message bus
        self.message_bus = RedisMessageBus()
        self.subscriber = MessageBusSubscriber(self.message_bus)
        self.publisher = MessageBusPublisher(self.message_bus)
        
        # Service state
        self.is_running = False
        self.fusion_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.fused_results_count = 0
        self.total_fusion_time = 0.0
        
        # Fusion parameters
        self.iou_threshold = 0.5  # IoU threshold for box matching
        self.confidence_boost = 0.1  # Confidence boost for VLM-confirmed detections
        
    async def start(self) -> None:
        """Start the fusion service."""
        try:
            self.logger.info("Starting fusion service")
            
            # Initialize message bus
            await self.subscriber.start()
            await self.publisher.start()
            
            # Subscribe to detection results
            await self.subscriber.subscribe_to_messages(
                Channels.YOLO_RESULTS,
                self._handle_yolo_result
            )
            
            await self.subscriber.subscribe_to_messages(
                Channels.VLM_RESULTS,
                self._handle_vlm_result
            )
            
            # Subscribe to camera frames for metadata
            await self.subscriber.subscribe_to_frames(
                Channels.CAMERA_FRAMES,
                self._handle_frame_metadata
            )
            
            # Start periodic fusion task
            self.fusion_task = asyncio.create_task(self._fusion_loop())
            
            self.is_running = True
            
            self.logger.info("Fusion service started successfully")
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="fusion_service_start")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the fusion service."""
        self.logger.info("Stopping fusion service")
        self.is_running = False
        
        # Cancel fusion task
        if self.fusion_task:
            self.fusion_task.cancel()
            try:
                await self.fusion_task
            except asyncio.CancelledError:
                pass
        
        # Stop message bus
        try:
            await self.subscriber.stop()
            await self.publisher.stop()
        except Exception as e:
            log_error_with_context(self.logger, e, operation="message_bus_stop")
        
        self.logger.info(f"Fusion service stopped. Fused {self.fused_results_count} results")
    
    async def _handle_yolo_result(self, message: DetectionMessage) -> None:
        """Handle incoming YOLO detection result."""
        try:
            self.cache.add_yolo_result(message.frame_id, message.result)
            
            self.logger.debug(
                f"Cached YOLO result for frame {message.frame_id}",
                extra={
                    "frame_id": message.frame_id,
                    "detections": len(message.result.bounding_boxes)
                }
            )
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"frame_id": message.frame_id},
                "handle_yolo_result"
            )
    
    async def _handle_vlm_result(self, message: VLMMessage) -> None:
        """Handle incoming VLM result."""
        try:
            self.cache.add_vlm_result(message.frame_id, message.result)
            
            self.logger.debug(
                f"Cached VLM result for frame {message.frame_id}",
                extra={
                    "frame_id": message.frame_id,
                    "caption_length": len(message.result.caption or ""),
                    "objects": len(message.result.objects)
                }
            )
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"frame_id": message.frame_id},
                "handle_vlm_result"
            )
    
    async def _handle_frame_metadata(self, frame, metadata: FrameMetadata) -> None:
        """Handle incoming frame metadata."""
        try:
            self.cache.add_frame_metadata(metadata.frame_id, metadata)
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"frame_id": metadata.frame_id},
                "handle_frame_metadata"
            )
    
    async def _fusion_loop(self) -> None:
        """Main fusion loop that periodically creates fused results."""
        while self.is_running:
            try:
                await self._create_fused_results()
                await asyncio.sleep(0.1)  # Check for fusion opportunities frequently
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error_with_context(self.logger, e, operation="fusion_loop")
                await asyncio.sleep(1.0)
    
    async def _create_fused_results(self) -> None:
        """Create fused results from available YOLO and VLM data."""
        try:
            start_time = time.perf_counter()
            
            # Get latest YOLO result
            latest_yolo = self.cache.get_latest_yolo_result()
            if not latest_yolo:
                return
            
            yolo_frame_id, yolo_result = latest_yolo
            
            # Try to find corresponding VLM result (same frame or recent)
            vlm_result = self.cache.get_vlm_result(yolo_frame_id)
            if not vlm_result:
                # Use the most recent VLM result as context
                latest_vlm = self.cache.get_latest_vlm_result()
                if latest_vlm:
                    _, vlm_result = latest_vlm
            
            # Get frame metadata
            frame_metadata = self.cache.get_frame_metadata(yolo_frame_id)
            if not frame_metadata:
                return
            
            # Create fused result
            fused_result = await self._fuse_results(
                frame_metadata, yolo_result, vlm_result
            )
            
            # Create fusion result message
            fusion_message = FusionResultMessage(
                result=fused_result,
                source_service="fusion_service"
            )
            
            # Publish fused result
            await self.publisher.publish_message(
                Channels.FUSION_RESULTS,
                fusion_message
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            self.total_fusion_time += processing_time
            self.fused_results_count += 1
            
            # Log performance metrics
            if (self.config.enable_performance_metrics and 
                self.fused_results_count % 10 == 0):
                avg_fusion_time = self.total_fusion_time / self.fused_results_count
                log_performance_metrics(
                    self.logger,
                    "result_fusion",
                    processing_time,
                    frame_id=yolo_frame_id,
                    model_name="fusion_service",
                    yolo_detections=len(yolo_result.bounding_boxes),
                    vlm_available=vlm_result is not None,
                    avg_fusion_time=avg_fusion_time
                )
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="create_fused_results")
    
    async def _fuse_results(
        self,
        frame_metadata: FrameMetadata,
        yolo_result: DetectionResult,
        vlm_result: Optional[VLMResult]
    ) -> ProcessingPipelineResult:
        """Fuse YOLO and VLM results into a unified result."""
        try:
            # Start with YOLO detections
            enhanced_boxes = list(yolo_result.bounding_boxes)
            
            if vlm_result:
                # Enhance YOLO detections with VLM insights
                enhanced_boxes = self._enhance_detections(
                    yolo_result.bounding_boxes,
                    vlm_result
                )
                
                # Add VLM-only detections that don't overlap with YOLO
                vlm_only_boxes = self._get_vlm_only_detections(
                    yolo_result.bounding_boxes,
                    vlm_result.bounding_boxes
                )
                enhanced_boxes.extend(vlm_only_boxes)
            
            # Create enhanced detection result
            enhanced_detection = DetectionResult(
                bounding_boxes=enhanced_boxes,
                processing_time_ms=yolo_result.processing_time_ms,
                model_name=f"fused_{yolo_result.model_name}",
                timestamp=yolo_result.timestamp
            )
            
            # Calculate total processing time
            total_time = yolo_result.processing_time_ms
            if vlm_result:
                total_time += vlm_result.processing_time_ms
            
            # Create fused result
            pipeline_result = ProcessingPipelineResult(
                frame_metadata=frame_metadata,
                yolo_result=yolo_result,
                vlm_result=vlm_result,
                total_processing_time_ms=total_time
            )
            
            return pipeline_result
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"frame_id": frame_metadata.frame_id},
                "fuse_results"
            )
            # Return basic result on error
            return ProcessingPipelineResult(
                frame_metadata=frame_metadata,
                yolo_result=yolo_result,
                vlm_result=vlm_result,
                total_processing_time_ms=yolo_result.processing_time_ms
            )
    
    def _enhance_detections(
        self,
        yolo_boxes: List[BoundingBox],
        vlm_result: VLMResult
    ) -> List[BoundingBox]:
        """Enhance YOLO detections with VLM insights."""
        enhanced_boxes = []
        
        for yolo_box in yolo_boxes:
            enhanced_box = BoundingBox(
                x1=yolo_box.x1,
                y1=yolo_box.y1,
                x2=yolo_box.x2,
                y2=yolo_box.y2,
                confidence=yolo_box.confidence,
                class_name=yolo_box.class_name,
                class_id=yolo_box.class_id
            )
            
            # Check if VLM mentioned this object type
            if self._vlm_mentions_object(vlm_result, yolo_box.class_name):
                # Boost confidence for VLM-confirmed detections
                enhanced_box.confidence = min(1.0, enhanced_box.confidence + self.confidence_boost)
            
            # Check for VLM bounding boxes that match this YOLO detection
            matching_vlm_box = self._find_matching_vlm_box(yolo_box, vlm_result.bounding_boxes)
            if matching_vlm_box:
                # Use VLM class name if different (potentially more specific)
                if matching_vlm_box.class_name != yolo_box.class_name:
                    enhanced_box.class_name = f"{yolo_box.class_name}({matching_vlm_box.class_name})"
            
            enhanced_boxes.append(enhanced_box)
        
        return enhanced_boxes
    
    def _vlm_mentions_object(self, vlm_result: VLMResult, object_class: str) -> bool:
        """Check if VLM mentions a specific object class."""
        # Check in objects list
        if object_class.lower() in [obj.lower() for obj in vlm_result.objects]:
            return True
        
        # Check in caption and description
        text_to_check = []
        if vlm_result.caption:
            text_to_check.append(vlm_result.caption.lower())
        if vlm_result.description:
            text_to_check.append(vlm_result.description.lower())
        
        for text in text_to_check:
            if object_class.lower() in text:
                return True
        
        return False
    
    def _find_matching_vlm_box(
        self,
        yolo_box: BoundingBox,
        vlm_boxes: List[BoundingBox]
    ) -> Optional[BoundingBox]:
        """Find VLM bounding box that matches a YOLO box."""
        best_match = None
        best_iou = 0.0
        
        for vlm_box in vlm_boxes:
            iou = self._calculate_iou(yolo_box, vlm_box)
            if iou > self.iou_threshold and iou > best_iou:
                best_iou = iou
                best_match = vlm_box
        
        return best_match
    
    def _get_vlm_only_detections(
        self,
        yolo_boxes: List[BoundingBox],
        vlm_boxes: List[BoundingBox]
    ) -> List[BoundingBox]:
        """Get VLM detections that don't overlap significantly with YOLO."""
        vlm_only = []
        
        for vlm_box in vlm_boxes:
            has_overlap = False
            for yolo_box in yolo_boxes:
                iou = self._calculate_iou(vlm_box, yolo_box)
                if iou > self.iou_threshold:
                    has_overlap = True
                    break
            
            if not has_overlap:
                # Reduce confidence for VLM-only detections
                vlm_only_box = BoundingBox(
                    x1=vlm_box.x1,
                    y1=vlm_box.y1,
                    x2=vlm_box.x2,
                    y2=vlm_box.y2,
                    confidence=vlm_box.confidence * 0.8,  # Reduce confidence
                    class_name=f"vlm_{vlm_box.class_name}",
                    class_id=vlm_box.class_id
                )
                vlm_only.append(vlm_only_box)
        
        return vlm_only
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        try:
            # Calculate intersection
            x1 = max(box1.x1, box2.x1)
            y1 = max(box1.y1, box2.y1)
            x2 = min(box1.x2, box2.x2)
            y2 = min(box1.y2, box2.y2)
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            
            # Calculate union
            area1 = box1.area()
            area2 = box2.area()
            union = area1 + area2 - intersection
            
            if union <= 0:
                return 0.0
            
            return intersection / union
            
        except Exception:
            return 0.0
    
    async def get_service_info(self) -> dict:
        """Get service information and statistics."""
        try:
            avg_fusion_time = (
                self.total_fusion_time / self.fused_results_count 
                if self.fused_results_count > 0 else 0.0
            )
            
            return {
                "service_name": "fusion_service",
                "is_running": self.is_running,
                "fused_results_count": self.fused_results_count,
                "average_fusion_time_ms": avg_fusion_time,
                "iou_threshold": self.iou_threshold,
                "confidence_boost": self.confidence_boost,
                "cache_stats": {
                    "yolo_results": len(self.cache.yolo_results),
                    "vlm_results": len(self.cache.vlm_results),
                    "frame_metadata": len(self.cache.frame_metadata)
                }
            }
        except Exception as e:
            log_error_with_context(self.logger, e, operation="get_service_info")
            return {}


async def main():
    """Main function for running the fusion service standalone."""
    service = FusionService()
    
    try:
        await service.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nShutting down fusion service...")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
