#!/usr/bin/env python3
"""
Native Moondream service for macOS that interfaces with the existing 'moondream' CLI.
Runs outside of containers to access Apple Silicon GPU.
"""

import asyncio
import os
import sys
import time
import logging
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import redis
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings

# Import shared models from the containerized services
from services.shared.models import (
    FrameMetadata, BoundingBox, VLMResult, 
    VLMMessage, FrameMessage, ChatRequest, ChatResponse
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
    """Native Moondream service using the existing CLI."""
    
    def __init__(self):
        self.config = NativeMoondreamConfig()
        self.setup_logging()
        
        self.redis_client: Optional[redis.Redis] = None
        self.frame_pubsub: Optional[redis.client.PubSub] = None
        self.chat_pubsub: Optional[redis.client.PubSub] = None
        
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
        
    def test_moondream_cli(self) -> bool:
        """Test if the moondream CLI is available."""
        try:
            result = subprocess.run(
                [self.config.moondream_cli_command, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.logger.info("Moondream CLI is available and working")
                return True
            else:
                self.logger.error(f"Moondream CLI test failed: {result.stderr}")
                return False
        except FileNotFoundError:
            self.logger.error(f"Moondream CLI not found: {self.config.moondream_cli_command}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("Moondream CLI test timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error testing Moondream CLI: {e}")
            return False
            
    def connect_redis(self):
        """Connect to Redis message bus."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")
            
            # Setup pub/sub for frames
            self.frame_pubsub = self.redis_client.pubsub()
            self.frame_pubsub.subscribe("camera.frames")
            self.logger.info("Subscribed to camera.frames channel")
            
            # Setup pub/sub for chat requests
            self.chat_pubsub = self.redis_client.pubsub()
            self.chat_pubsub.subscribe("chat.requests")
            self.logger.info("Subscribed to chat.requests channel")
            
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
            
    def process_frame(self, frame_data: dict) -> Optional[VLMMessage]:
        """Process a single frame with Moondream VLM."""
        try:
            start_time = time.perf_counter()
            
            # Decode frame metadata
            frame_metadata = FrameMetadata(**frame_data)
            
            # Skip frames based on stride
            if self.frames_processed % self.config.vlm_frame_stride != 0:
                return None
                
            # Create temporary image file (in real implementation, decode from Redis)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                # In production, you'd write the actual image data here
                # For now, create a placeholder
                tmp_path = tmp_file.name
                
            try:
                # Call Moondream CLI
                description = self.call_moondream_cli(tmp_path, "Describe this image in detail")
                
                if description:
                    # Create VLM result
                    vlm_result = VLMResult(
                        description=description,
                        confidence=0.95,  # CLI doesn't provide confidence, use default
                        detected_objects=[],  # CLI output would need parsing for objects
                        bounding_boxes=[]  # CLI doesn't provide bounding boxes by default
                    )
                    
                    processing_time = (time.perf_counter() - start_time) * 1000
                    self.total_processing_time += processing_time
                    self.frames_processed += 1
                    
                    vlm_message = VLMMessage(
                        frame_id=frame_metadata.frame_id,
                        timestamp=frame_metadata.timestamp,
                        vlm_result=vlm_result,
                        processing_time_ms=processing_time,
                        model_name="moondream2",
                        prompt="Describe this image in detail"
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
            
    def process_chat_request(self, chat_data: dict) -> Optional[ChatResponse]:
        """Process a chat request with Moondream."""
        try:
            start_time = time.perf_counter()
            
            # Parse chat request
            chat_request = ChatRequest(**chat_data)
            
            # Create temporary image file if image is provided
            tmp_path = None
            if hasattr(chat_request, 'image_data') and chat_request.image_data:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    # In production, decode base64 image data
                    tmp_path = tmp_file.name
                    
            try:
                # Call Moondream CLI with the user's prompt
                if tmp_path:
                    response_text = self.call_moondream_cli(tmp_path, chat_request.message)
                else:
                    # Text-only request - use a default image or handle differently
                    response_text = "I need an image to analyze. Please provide an image with your question."
                
                if response_text:
                    processing_time = (time.perf_counter() - start_time) * 1000
                    self.chat_requests_processed += 1
                    
                    chat_response = ChatResponse(
                        request_id=chat_request.request_id,
                        response=response_text,
                        processing_time_ms=processing_time,
                        model_name="moondream2",
                        timestamp=time.time()
                    )
                    
                    return chat_response
                    
            finally:
                # Clean up temporary file
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                        
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing chat request: {e}")
            return None
            
    def publish_vlm_result(self, vlm_message: VLMMessage):
        """Publish VLM results to Redis."""
        try:
            message_dict = vlm_message.model_dump()
            self.redis_client.publish("detection.vlm", str(message_dict))
        except Exception as e:
            self.logger.error(f"Failed to publish VLM result: {e}")
            
    def publish_chat_response(self, chat_response: ChatResponse):
        """Publish chat response to Redis."""
        try:
            message_dict = chat_response.model_dump()
            self.redis_client.publish("chat.responses", str(message_dict))
        except Exception as e:
            self.logger.error(f"Failed to publish chat response: {e}")
            
    async def run(self):
        """Main service loop."""
        self.logger.info("Starting Native Moondream service")
        
        try:
            # Test Moondream CLI
            if not self.test_moondream_cli():
                raise RuntimeError("Moondream CLI is not available")
                
            # Connect to Redis
            self.connect_redis()
            
            self.logger.info("Native Moondream service started successfully")
            
            # Process messages from both channels
            while True:
                # Check frame messages
                try:
                    frame_message = self.frame_pubsub.get_message(timeout=0.1)
                    if frame_message and frame_message['type'] == 'message':
                        frame_data = eval(frame_message['data'])
                        vlm_result = self.process_frame(frame_data)
                        if vlm_result:
                            self.publish_vlm_result(vlm_result)
                except Exception as e:
                    self.logger.error(f"Error processing frame message: {e}")
                
                # Check chat messages
                try:
                    chat_message = self.chat_pubsub.get_message(timeout=0.1)
                    if chat_message and chat_message['type'] == 'message':
                        chat_data = eval(chat_message['data'])
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
