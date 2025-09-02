"""Moondream VLM service with parallel instance support."""

import asyncio
import os
import time
import numpy as np
from typing import Optional, List, Dict, Any, Union
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

from ..message_bus.redis_bus import RedisMessageBus
from ..message_bus.base import MessageBusSubscriber, MessageBusPublisher, Channels
from ..shared.models import (
    FrameMetadata, BoundingBox, VLMResult, Point,
    VLMMessage, ChatMessage, ChatResponse, ChatRequestMessage, ChatResponseMessage
)
from ..shared.logging_config import setup_logging, log_performance_metrics, log_error_with_context
from ..shared.config import get_config


class MoondreamInstance:
    """Individual Moondream model instance."""
    
    def __init__(self, instance_id: int, config, logger):
        self.instance_id = instance_id
        self.config = config
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.device = config.moondream_device
        self.is_loaded = False
        
    async def load_model(self) -> None:
        """Load the Moondream model."""
        try:
            self.logger.info(f"Loading Moondream instance {self.instance_id}")
            
            # Load in thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _load():
                # Use local cache directory if available
                cache_dir = "/app/models/moondream" if os.path.exists("/app/models/moondream") else None
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.moondream_model,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map=None,  # We'll handle device placement manually
                    cache_dir=cache_dir
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.moondream_model,
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
                return model, tokenizer
            
            self.model, self.tokenizer = await loop.run_in_executor(None, _load)
            
            # Move to device
            if self.device == "mps" and torch.backends.mps.is_available():
                self.model = self.model.to("mps")
                self.logger.info(f"Instance {self.instance_id} loaded on Apple Silicon MPS")
            elif self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                self.logger.info(f"Instance {self.instance_id} loaded on CUDA")
            else:
                self.device = "cpu"
                self.logger.info(f"Instance {self.instance_id} loaded on CPU")
            
            self.is_loaded = True
            self.logger.info(f"Moondream instance {self.instance_id} loaded successfully")
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"instance_id": self.instance_id, "model": self.config.moondream_model},
                "model_loading"
            )
            raise
    
    async def unload_model(self) -> None:
        """Unload the model to free memory."""
        try:
            if self.model:
                del self.model
                self.model = None
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear GPU cache
            if self.device in ["cuda", "mps"]:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                elif hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
            self.is_loaded = False
            self.logger.info(f"Moondream instance {self.instance_id} unloaded")
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"instance_id": self.instance_id},
                "model_unloading"
            )
    
    async def generate_caption(self, image: Image.Image) -> str:
        """Generate a caption for the image."""
        if not self.is_loaded:
            raise RuntimeError(f"Instance {self.instance_id} not loaded")
        
        try:
            loop = asyncio.get_event_loop()
            
            def _generate():
                # Generate caption
                enc_image = self.model.encode_image(image)
                return self.model.answer_question(enc_image, "Describe this image.", self.tokenizer)
            
            caption = await loop.run_in_executor(None, _generate)
            return caption.strip()
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"instance_id": self.instance_id},
                "caption_generation"
            )
            return ""
    
    async def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about the image."""
        if not self.is_loaded:
            raise RuntimeError(f"Instance {self.instance_id} not loaded")
        
        try:
            loop = asyncio.get_event_loop()
            
            def _answer():
                enc_image = self.model.encode_image(image)
                return self.model.answer_question(enc_image, question, self.tokenizer)
            
            answer = await loop.run_in_executor(None, _answer)
            return answer.strip()
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"instance_id": self.instance_id, "question": question},
                "question_answering"
            )
            return ""
    
    async def detect_objects(self, image: Image.Image, query: str = None) -> List[BoundingBox]:
        """Detect objects in the image (if supported by the model)."""
        if not self.is_loaded:
            raise RuntimeError(f"Instance {self.instance_id} not loaded")
        
        # Note: This is a placeholder for object detection
        # The actual implementation depends on the specific Moondream model capabilities
        try:
            if query is None:
                query = "What objects do you see in this image?"
            
            description = await self.answer_question(image, query)
            
            # Extract potential object mentions (simple heuristic)
            # This would need to be enhanced based on actual model capabilities
            objects = self._extract_objects_from_description(description)
            
            return objects
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"instance_id": self.instance_id, "query": query},
                "object_detection"
            )
            return []
    
    def _extract_objects_from_description(self, description: str) -> List[BoundingBox]:
        """Extract object mentions from description (placeholder implementation)."""
        # This is a simple heuristic - in practice, you'd want to use
        # the model's actual object detection capabilities if available
        objects = []
        
        # Look for common object patterns
        object_patterns = [
            r'\b(person|people|man|woman|child)\b',
            r'\b(car|vehicle|truck|bike|bicycle)\b',
            r'\b(dog|cat|animal|bird)\b',
            r'\b(chair|table|desk|sofa|bed)\b',
            r'\b(phone|laptop|computer|tv|television)\b'
        ]
        
        for pattern in object_patterns:
            matches = re.findall(pattern, description.lower())
            for match in matches:
                # Create placeholder bounding box (would need actual coordinates)
                bbox = BoundingBox(
                    x1=0.0, y1=0.0, x2=100.0, y2=100.0,
                    confidence=0.5,
                    class_name=match,
                    class_id=None
                )
                objects.append(bbox)
        
        return objects


class MoondreamService:
    """Moondream VLM service with parallel instance support."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("moondream_service")
        
        # Model instances
        self.instances: List[MoondreamInstance] = []
        self.instance_queue: Optional[asyncio.Queue] = None
        
        # Message bus
        self.message_bus = RedisMessageBus()
        self.subscriber = MessageBusSubscriber(self.message_bus)
        self.publisher = MessageBusPublisher(self.message_bus)
        
        # Service state
        self.is_running = False
        
        # Performance tracking
        self.frames_processed = 0
        self.total_processing_time = 0.0
        self.chat_requests_processed = 0
        
        # Frame processing control
        self.frame_stride = self.config.vlm_frame_stride
        self.frame_counter = 0
        
    async def start(self) -> None:
        """Start the Moondream service."""
        try:
            self.logger.info("Starting Moondream service")
            
            # Load model instances
            await self._load_instances()
            
            # Initialize message bus
            await self.subscriber.start()
            await self.publisher.start()
            
            # Subscribe to camera frames for periodic processing
            await self.subscriber.subscribe_to_frames(
                Channels.CAMERA_FRAMES,
                self._process_frame
            )
            
            # Subscribe to chat requests
            await self.subscriber.subscribe_to_messages(
                Channels.CHAT_REQUESTS,
                self._process_chat_request
            )
            
            self.is_running = True
            
            self.logger.info(
                "Moondream service started successfully",
                extra={
                    "model": self.config.moondream_model,
                    "instances": len(self.instances),
                    "device": self.config.moondream_device,
                    "frame_stride": self.frame_stride
                }
            )
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="moondream_service_start")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the Moondream service."""
        self.logger.info("Stopping Moondream service")
        self.is_running = False
        
        # Stop message bus
        try:
            await self.subscriber.stop()
            await self.publisher.stop()
        except Exception as e:
            log_error_with_context(self.logger, e, operation="message_bus_stop")
        
        # Unload model instances
        await self._unload_instances()
        
        self.logger.info(
            f"Moondream service stopped. Processed {self.frames_processed} frames, "
            f"{self.chat_requests_processed} chat requests"
        )
    
    async def _load_instances(self) -> None:
        """Load multiple Moondream instances."""
        try:
            num_instances = self.config.moondream_instances
            self.logger.info(f"Loading {num_instances} Moondream instances")
            
            # Create instances
            for i in range(num_instances):
                instance = MoondreamInstance(i, self.config, self.logger)
                self.instances.append(instance)
            
            # Load models in parallel
            load_tasks = [instance.load_model() for instance in self.instances]
            await asyncio.gather(*load_tasks)
            
            # Create instance queue for load balancing
            self.instance_queue = asyncio.Queue()
            for instance in self.instances:
                await self.instance_queue.put(instance)
            
            self.logger.info(f"Loaded {len(self.instances)} Moondream instances")
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"num_instances": self.config.moondream_instances},
                "load_instances"
            )
            raise
    
    async def _unload_instances(self) -> None:
        """Unload all model instances."""
        try:
            unload_tasks = [instance.unload_model() for instance in self.instances]
            await asyncio.gather(*unload_tasks, return_exceptions=True)
            self.instances.clear()
            self.instance_queue = None
        except Exception as e:
            log_error_with_context(self.logger, e, operation="unload_instances")
    
    async def _get_instance(self) -> MoondreamInstance:
        """Get an available instance."""
        return await self.instance_queue.get()
    
    async def _return_instance(self, instance: MoondreamInstance) -> None:
        """Return an instance to the queue."""
        await self.instance_queue.put(instance)
    
    async def _process_frame(self, frame: np.ndarray, metadata: FrameMetadata) -> None:
        """Process a frame with Moondream VLM."""
        try:
            self.frame_counter += 1
            
            # Skip frames based on stride
            if self.frame_counter % self.frame_stride != 0:
                return
            
            start_time = time.perf_counter()
            
            # Convert frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Get an available instance
            instance = await self._get_instance()
            
            try:
                # Generate caption
                caption = await instance.generate_caption(image)
                
                # Generate description with more detail
                description = await instance.answer_question(
                    image,
                    "Describe this image in detail, including objects, people, actions, and setting."
                )
                
                # Detect objects (if supported)
                objects = await instance.detect_objects(image)
                
                processing_time = (time.perf_counter() - start_time) * 1000
                self.total_processing_time += processing_time
                self.frames_processed += 1
                
                # Create VLM result
                vlm_result = VLMResult(
                    caption=caption,
                    description=description,
                    objects=[obj.class_name for obj in objects],
                    bounding_boxes=objects,
                    processing_time_ms=processing_time,
                    model_name=f"moondream_{self.config.moondream_model}",
                    timestamp=metadata.timestamp,
                    confidence=0.8  # Placeholder confidence
                )
                
                # Publish result
                vlm_message = VLMMessage(
                    source_service="moondream_service",
                    result=vlm_result,
                    frame_id=metadata.frame_id
                )
                
                await self.publisher.publish_message(
                    Channels.VLM_RESULTS,
                    vlm_message
                )
                
                # Log performance metrics
                if (self.config.enable_performance_metrics and 
                    self.frames_processed % 5 == 0):
                    avg_processing_time = self.total_processing_time / self.frames_processed
                    log_performance_metrics(
                        self.logger,
                        "vlm_inference",
                        processing_time,
                        frame_id=metadata.frame_id,
                        model_name=f"moondream_{self.config.moondream_model}",
                        instance_id=instance.instance_id,
                        avg_processing_time=avg_processing_time,
                        caption_length=len(caption),
                        description_length=len(description)
                    )
                
            finally:
                # Return instance to queue
                await self._return_instance(instance)
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"frame_id": metadata.frame_id},
                "frame_processing"
            )
    
    async def _process_chat_request(self, message: ChatRequestMessage) -> None:
        """Process a chat request."""
        try:
            start_time = time.perf_counter()
            chat_request = message.chat_message
            
            # Get the latest frame for context (placeholder - would need frame store)
            # For now, we'll respond without image context
            response_text = await self._generate_chat_response(chat_request)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            self.chat_requests_processed += 1
            
            # Create chat response
            chat_response = ChatResponse(
                response=response_text,
                processing_time_ms=processing_time,
                model_name=f"moondream_{self.config.moondream_model}",
                frame_id=chat_request.frame_id
            )
            
            # Publish response
            response_message = ChatResponseMessage(
                source_service="moondream_service",
                chat_response=chat_response
            )
            
            await self.publisher.publish_message(
                Channels.CHAT_RESPONSES,
                response_message
            )
            
            log_performance_metrics(
                self.logger,
                "chat_response",
                processing_time,
                model_name=f"moondream_{self.config.moondream_model}",
                message_length=len(chat_request.message),
                response_length=len(response_text)
            )
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"message": chat_request.message if 'chat_request' in locals() else "unknown"},
                "chat_processing"
            )
    
    async def _generate_chat_response(self, chat_request: ChatMessage) -> str:
        """Generate a response to a chat message."""
        # Get an available instance
        instance = await self._get_instance()
        
        try:
            # For now, generate a simple response
            # In a full implementation, you'd use the current frame as context
            response = f"I received your message: '{chat_request.message}'. "
            response += "To provide more specific answers, I would need to see the current camera view."
            
            return response
            
        finally:
            await self._return_instance(instance)
    
    async def get_service_info(self) -> dict:
        """Get service information and statistics."""
        try:
            avg_processing_time = (
                self.total_processing_time / self.frames_processed 
                if self.frames_processed > 0 else 0.0
            )
            
            return {
                "service_name": "moondream_service",
                "model": self.config.moondream_model,
                "device": self.config.moondream_device,
                "instances": len(self.instances),
                "is_running": self.is_running,
                "frames_processed": self.frames_processed,
                "chat_requests_processed": self.chat_requests_processed,
                "average_processing_time_ms": avg_processing_time,
                "frame_stride": self.frame_stride,
                "instances_loaded": sum(1 for inst in self.instances if inst.is_loaded)
            }
        except Exception as e:
            log_error_with_context(self.logger, e, operation="get_service_info")
            return {}


async def main():
    """Main function for running the Moondream service standalone."""
    service = MoondreamService()
    
    try:
        await service.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nShutting down Moondream service...")
    finally:
        await service.stop()


if __name__ == "__main__":
    # Add missing import
    import cv2
    asyncio.run(main())
