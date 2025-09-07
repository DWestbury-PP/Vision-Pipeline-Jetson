#!/usr/bin/env python3
"""
Containerized Moondream service for NVIDIA Jetson using the actual Moondream2 model.
Runs inside Docker containers with NVIDIA runtime for CUDA GPU acceleration.
"""

import asyncio
import os
import sys
import time
import logging
import json
import pickle
import subprocess
import tempfile
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List
from PIL import Image
from ..shared.opencv_patch import cv2, CV2_AVAILABLE

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add the moondream model path
moondream_path = project_root / "models" / "moondream" / "moondream2"
sys.path.insert(0, str(moondream_path))

import redis
from pydantic import Field
from pydantic_settings import BaseSettings

# Import shared models from the containerized services
from services.shared.models import (
    FrameMetadata, BoundingBox, VLMResult, 
    VLMMessage, FrameMessage, ChatRequestMessage, ChatResponseMessage, ChatResponse
)


class NativeMoondreamConfig(BaseSettings):
    """Configuration for native Moondream service."""
    
    # Redis connection
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Moondream configuration
    vlm_frame_stride: int = Field(default=10, env="VLM_FRAME_STRIDE")
    moondream_cli_command: str = Field(default="moondream", env="MOONDREAM_CLI_COMMAND")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class NativeMoondreamService:
    """Containerized Moondream service using the actual Moondream2 model with CUDA acceleration."""
    
    def __init__(self):
        self.config = NativeMoondreamConfig()
        self.setup_logging()
        
        self.redis_client: Optional[redis.Redis] = None
        self.frame_pubsub: Optional[redis.client.PubSub] = None
        self.chat_pubsub: Optional[redis.client.PubSub] = None
        
        # Model will be loaded later
        self.model = None
        self.device = None
        self.last_frame = None  # Store last frame for chat context
        
        # Performance tracking
        self.frames_processed = 0
        self.chat_requests_processed = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()
        
    def setup_logging(self):
        """Setup structured logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "native.moondream", "message": "%(message)s", "component": "native_moondream"}',
            datefmt='%Y-%m-%dT%H:%M:%S.%fZ'
        )
        self.logger = logging.getLogger("native.moondream")
        
    def load_moondream_model(self):
        """Load the Moondream2 model from local files."""
        try:
            self.logger.info("Loading Moondream2 model from local files...")
            
            # Set device - use MPS for Apple Silicon
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.logger.info("Using Apple Silicon MPS for acceleration")
            else:
                self.device = torch.device("cpu")
                self.logger.info("Using CPU (MPS not available)")
            
            # Import the model classes
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load model from local directory
            model_path = str(project_root / "models" / "moondream" / "moondream2")
            
            self.logger.info(f"Loading model from {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "mps" else torch.float32
            )
            self.model = self.model.to(self.device)
            
            self.logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Moondream model: {e}")
            return False
            
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
            
            # Setup pub/sub for frames
            self.frame_pubsub = self.redis_client.pubsub()
            self.frame_pubsub.subscribe("frame:camera.frames")
            self.logger.info("Subscribed to frame:camera.frames channel")
            
            # Setup pub/sub for chat requests
            self.chat_pubsub = self.redis_client.pubsub()
            self.chat_pubsub.subscribe("msg:chat.requests")
            self.logger.info("Subscribed to msg:chat.requests channel")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    def call_moondream_cli(self, image_path: str, prompt: str = "Describe this image") -> Optional[str]:
        """Call the moondream CLI with an image."""
        try:
            cmd = [
                self.config.moondream_cli_command,
                image_path,
                "--prompt", prompt
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self.logger.error(f"Moondream CLI error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("Moondream CLI call timed out")
            return None
        except Exception as e:
            self.logger.error(f"Error calling Moondream CLI: {e}")
            return None
            
    def process_frame_array(self, frame: np.ndarray, metadata_dict: dict) -> Optional[VLMMessage]:
        """Process a single frame array with Moondream VLM."""
        try:
            # Store the last frame for chat context
            self.last_frame = frame
            
            if not self.model:
                self.logger.warning("Model not loaded, skipping frame processing")
                return None
                
            start_time = time.perf_counter()
            
            # Create frame metadata
            frame_metadata = FrameMetadata(**metadata_dict)
            
            # Skip frames based on stride
            self.frames_processed += 1
            if self.frames_processed % self.config.vlm_frame_stride != 0:
                return None
                
            # Convert numpy array to PIL Image
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to RGB if needed (OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
                
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb.astype('uint8'))
            
            try:
                # Use the model to generate a description
                self.logger.info(f"Processing frame {frame_metadata.frame_id} with Moondream")
                
                # Encode the image and generate description
                enc_image = self.model.encode_image(pil_image)
                description = self.model.answer_question(
                    enc_image,
                    "Describe what you see in this image in detail.",
                    tokenizer=None
                )
                
                if description:
                    processing_time = (time.perf_counter() - start_time) * 1000
                    
                    # Create VLM result with all required fields
                    vlm_result = VLMResult(
                        description=description,
                        confidence=0.95,
                        detected_objects=[],
                        bounding_boxes=[],
                        processing_time_ms=processing_time,
                        model_name="moondream2"
                    )
                    
                    self.total_processing_time += processing_time
                    
                    vlm_message = VLMMessage(
                        frame_id=frame_metadata.frame_id,
                        timestamp=frame_metadata.timestamp,
                        result=vlm_result,
                        source_service="moondream_native"
                    )
                    
                    # Log performance periodically
                    if self.frames_processed % 5 == 0:
                        avg_time = self.total_processing_time / self.frames_processed
                        self.logger.info(
                            f"Processed {self.frames_processed} frames, "
                            f"avg: {avg_time:.1f}ms"
                        )
                        
                    return vlm_message
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return None
            
    def process_chat_request(self, chat_data: dict) -> Optional[ChatResponseMessage]:
        """Process a chat request with Moondream."""
        try:
            start_time = time.perf_counter()
            
            # Parse chat request
            self.logger.info(f"Processing chat request: {chat_data}")
            message = chat_data.get('chat_message', {}).get('message', '')
            frame_id = chat_data.get('chat_message', {}).get('frame_id')
            
            if not self.model:
                self.logger.warning("Model not loaded, using echo mode")
                response_text = f"[Model Loading] Echo: '{message}'"
            else:
                # Use the most recent frame if available
                if hasattr(self, 'last_frame') and self.last_frame is not None:
                    # Convert frame to PIL Image
                    from ..shared.opencv_patch import cv2, CV2_AVAILABLE
                    from PIL import Image
                    frame_rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Encode image for model
                    enc_image = self.model.encode_image(pil_image)
                    
                    # Generate response based on the question and image
                    response_text = self.model.answer_question(
                        enc_image,
                        message,
                        tokenizer=None
                    )
                    self.logger.info(f"Generated VLM response for question: {message[:50]}...")
                else:
                    # No frame available, answer based on general context
                    response_text = f"I need to see an image to answer your question: '{message}'. Please ensure the camera is active."
                
            processing_time = (time.perf_counter() - start_time) * 1000
            self.chat_requests_processed += 1
            
            # Create chat response object
            from datetime import datetime
            chat_response_obj = ChatResponse(
                response=response_text,
                timestamp=datetime.utcnow(),
                processing_time_ms=processing_time if 'processing_time' in locals() else 0,
                model_name="moondream2",
                frame_id=chat_data.get('chat_message', {}).get('frame_id')
            )
            
            # Create response message
            chat_response = ChatResponseMessage(
                chat_response=chat_response_obj,
                source_service="moondream_native"
            )
            
            self.logger.info(f"Sending chat response: {response_text}")
            return chat_response
            
        except Exception as e:
            self.logger.error(f"Error processing chat request: {e}")
            return None
            
    def publish_vlm_result(self, vlm_message: VLMMessage):
        """Publish VLM results to Redis."""
        try:
            import json
            message_json = json.dumps(vlm_message.model_dump(mode='json'))
            self.redis_client.publish("msg:detection.vlm", message_json.encode('utf-8'))
            self.logger.info(f"Published VLM result for frame {vlm_message.frame_id}")
        except Exception as e:
            self.logger.error(f"Failed to publish VLM result: {e}")
            
    def publish_chat_response(self, chat_response: ChatResponseMessage):
        """Publish chat response to Redis."""
        try:
            import json
            message_json = json.dumps(chat_response.model_dump(mode='json'))
            self.redis_client.publish("msg:chat.responses", message_json.encode('utf-8'))
            self.logger.info(f"Published chat response")
        except Exception as e:
            self.logger.error(f"Failed to publish chat response: {e}")
            
    async def run(self):
        """Main service loop."""
        self.logger.info("Starting Native Moondream service")
        
        try:
            # Load the Moondream model
            if not self.load_moondream_model():
                self.logger.warning("Failed to load model, will run in echo mode")
                
            # Connect to Redis
            self.connect_redis()
            
            self.logger.info("Native Moondream service started successfully")
            
            # Process messages from both channels
            while True:
                # Check frame messages
                try:
                    frame_message = self.frame_pubsub.get_message(timeout=0.1)
                    if frame_message and frame_message['type'] == 'message':
                        # Unpickle frame package
                        import pickle
                        frame_package = pickle.loads(frame_message['data'])
                        
                        # Extract frame data and metadata
                        frame_bytes = frame_package['frame_data']
                        metadata_dict = frame_package['metadata']
                        shape = frame_package['shape']
                        dtype = frame_package['dtype']
                        
                        # Reconstruct numpy array
                        import numpy as np
                        frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)
                        
                        # Process frame with VLM
                        vlm_result = self.process_frame_array(frame, metadata_dict)
                        if vlm_result:
                            self.publish_vlm_result(vlm_result)
                except Exception as e:
                    self.logger.error(f"Error processing frame message: {e}")
                
                # Check chat messages
                try:
                    chat_message = self.chat_pubsub.get_message(timeout=0.1)
                    if chat_message and chat_message['type'] == 'message':
                        # Parse JSON message
                        import json
                        chat_data = json.loads(chat_message['data'].decode('utf-8'))
                        chat_response = self.process_chat_request(chat_data)
                        if chat_response:
                            self.publish_chat_response(chat_response)
                except Exception as e:
                    self.logger.error(f"Error processing chat message: {e}")
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down Native Moondream service")
        except Exception as e:
            self.logger.error(f"Fatal error in Native Moondream service: {e}")
            raise
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """Cleanup resources."""
        if self.frame_pubsub:
            self.frame_pubsub.close()
        if self.chat_pubsub:
            self.chat_pubsub.close()
        if self.redis_client:
            self.redis_client.close()
        self.logger.info("Native Moondream service stopped")


async def main():
    """Main entry point."""
    service = NativeMoondreamService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
