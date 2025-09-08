"""Redis-based message bus implementation for high-performance pub/sub."""

import asyncio
import json
import pickle
import gzip
from typing import Callable, Any, Optional, Dict
import numpy as np
import redis.asyncio as aioredis
from redis.asyncio.client import PubSub

from .base import MessageBus
from ..shared.models import BusMessage, FrameMetadata, FrameMessage
from ..shared.logging_config import setup_logging, log_error_with_context
from ..shared.config import get_config


class RedisMessageBus(MessageBus):
    """Redis-based message bus implementation with frame compression."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("redis_bus")
        self.redis_client: Optional[aioredis.Redis] = None
        self.pubsub: Optional[PubSub] = None
        self.frame_pubsub: Optional[PubSub] = None  # Separate pubsub for frames
        self.subscriptions: Dict[str, asyncio.Task] = {}
        self._connected = False
        
        # Frame compression settings
        self.compress_frames = True
        self.compression_level = 1  # Fast compression
        
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis_client = aioredis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                db=self.config.redis_db,
                decode_responses=False,  # We handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            self.pubsub = self.redis_client.pubsub()
            self.frame_pubsub = self.redis_client.pubsub()  # Separate for frames
            self._connected = True
            
            self.logger.info(
                f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}",
                extra={"redis_host": self.config.redis_host, "redis_port": self.config.redis_port}
            )
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"redis_host": self.config.redis_host, "redis_port": self.config.redis_port},
                "redis_connect"
            )
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        try:
            # Cancel all subscription tasks
            for task in self.subscriptions.values():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self.subscriptions.clear()
            
            if self.pubsub:
                await self.pubsub.close()
                self.pubsub = None
                
            if self.frame_pubsub:
                await self.frame_pubsub.close()
                self.frame_pubsub = None
                
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
                
            self._connected = False
            self.logger.info("Disconnected from Redis")
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="redis_disconnect")
    
    async def publish_message(self, channel: str, message: BusMessage) -> None:
        """Publish a message to a Redis channel."""
        if not self._connected:
            raise RuntimeError("Not connected to Redis")
            
        try:
            message_data = message.json().encode('utf-8')
            await self.redis_client.publish(f"msg:{channel}", message_data)
            
            self.logger.debug(
                f"Published message to channel {channel}",
                extra={"channel": channel, "message_type": message.message_type}
            )
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"channel": channel, "message_type": message.message_type},
                "publish_message"
            )
            raise
    
    async def publish_frame(
        self,
        channel: str,
        frame_data: np.ndarray,
        metadata: FrameMetadata
    ) -> None:
        """Publish frame data to a Redis channel with compression."""
        if not self._connected:
            raise RuntimeError("Not connected to Redis")
            
        try:
            # Serialize frame data
            frame_bytes = frame_data.tobytes()
            
            # Compress if enabled
            if self.compress_frames:
                frame_bytes = gzip.compress(frame_bytes, compresslevel=self.compression_level)
            
            # Create frame package
            frame_package = {
                'metadata': metadata.dict(),
                'frame_data': frame_bytes,
                'shape': frame_data.shape,
                'dtype': str(frame_data.dtype),
                'compressed': self.compress_frames
            }
            
            # Serialize package
            package_data = pickle.dumps(frame_package)
            
            # Publish frame data
            await self.redis_client.publish(f"frame:{channel}", package_data)
            
            # Also publish frame message for services that only need metadata
            frame_message = FrameMessage(
                source_service="camera",
                frame_metadata=metadata
            )
            await self.publish_message(channel, frame_message)
            
            self.logger.debug(
                f"Published frame to channel {channel}",
                extra={
                    "channel": channel,
                    "frame_id": metadata.frame_id,
                    "frame_size": len(package_data),
                    "compressed": self.compress_frames
                }
            )
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"channel": channel, "frame_id": metadata.frame_id},
                "publish_frame"
            )
            raise
    
    async def subscribe_message(
        self,
        channel: str,
        callback: Callable[[BusMessage], None]
    ) -> None:
        """Subscribe to messages on a Redis channel."""
        if not self._connected:
            raise RuntimeError("Not connected to Redis")
            
        try:
            # Create a separate pubsub instance for each subscription
            channel_pubsub = self.redis_client.pubsub()
            await channel_pubsub.subscribe(f"msg:{channel}")
            
            async def message_handler():
                try:
                    async for message in channel_pubsub.listen():
                        
                        if message['type'] == 'message' and message['channel'].decode('utf-8') == f"msg:{channel}":
                            try:
                                message_data = json.loads(message['data'].decode('utf-8'))
                            
                                # Reconstruct message object based on type
                                message_type = message_data.get('message_type')
                                if message_type:
                                    
                                    # Import message classes dynamically to avoid circular imports
                                    from ..shared.models import (
                                        DetectionMessage, VLMMessage, ChatRequestMessage,
                                        ChatResponseMessage, StatusMessage, ConfigurationMessage,
                                        FusionResultMessage
                                    )
                                    
                                    message_classes = {
                                        'detection': DetectionMessage,
                                        'vlm': VLMMessage,
                                        'chat_request': ChatRequestMessage,
                                        'chat_response': ChatResponseMessage,
                                        'status': StatusMessage,
                                        'configuration': ConfigurationMessage,
                                        'frame': FrameMessage,
                                        'fusion_result': FusionResultMessage
                                    }
                                    
                                    message_class = message_classes.get(message_type, BusMessage)
                                    # Use parse_obj for Pydantic 1.x compatibility
                                    if hasattr(message_class, 'model_validate'):
                                        parsed_message = message_class.model_validate(message_data)
                                    else:
                                        parsed_message = message_class.parse_obj(message_data)
                                    
                                    # Call async callback directly
                                    await callback(parsed_message)
                                    
                            except Exception as e:
                                log_error_with_context(
                                    self.logger,
                                    e,
                                    {"channel": channel, "message": str(message)},
                                    "message_handler"
                                )
                except Exception as e:
                    self.logger.error(f"Error in message handler for {channel}: {e}")
            
            # Start handler task
            task = asyncio.create_task(message_handler())
            self.subscriptions[f"msg:{channel}"] = task
            
            self.logger.info(
                f"Subscribed to messages on channel {channel}",
                extra={"channel": channel}
            )
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"channel": channel},
                "subscribe_message"
            )
            raise
    
    async def subscribe_frame(
        self,
        channel: str,
        callback: Callable[[np.ndarray, FrameMetadata], None]
    ) -> None:
        """Subscribe to frame data on a Redis channel."""
        if not self._connected:
            raise RuntimeError("Not connected to Redis")
            
        try:
            await self.frame_pubsub.subscribe(f"frame:{channel}")
            
            async def frame_handler():
                self.logger.info(f"Frame handler started for channel frame:{channel}")
                async for message in self.frame_pubsub.listen():
                    self.logger.info(f"Got message type: {message.get('type')}, channel: {message.get('channel')}")
                    if message['type'] == 'message' and message['channel'].decode('utf-8') == f"frame:{channel}":
                        try:
                            self.logger.info(f"Processing frame from channel frame:{channel}")
                            # Deserialize frame package
                            frame_package = pickle.loads(message['data'])
                            
                            # Extract metadata - Pydantic 1.x compatibility
                            if hasattr(FrameMetadata, 'model_validate'):
                                metadata = FrameMetadata.model_validate(frame_package['metadata'])
                            else:
                                metadata = FrameMetadata.parse_obj(frame_package['metadata'])
                            
                            # Reconstruct frame data
                            frame_bytes = frame_package['frame_data']
                            
                            # Decompress if needed
                            if frame_package.get('compressed', False):
                                frame_bytes = gzip.decompress(frame_bytes)
                            
                            # Reconstruct numpy array
                            frame_data = np.frombuffer(
                                frame_bytes,
                                dtype=frame_package['dtype']
                            ).reshape(frame_package['shape'])
                            
                            # Call async callback directly
                            self.logger.info(f"Calling callback for frame {metadata.frame_id}")
                            await callback(frame_data, metadata)
                            self.logger.info(f"Callback completed for frame {metadata.frame_id}")
                            
                        except Exception as e:
                            log_error_with_context(
                                self.logger,
                                e,
                                {"channel": channel},
                                "frame_handler"
                            )
            
            # Start handler task
            task = asyncio.create_task(frame_handler())
            self.subscriptions[f"frame:{channel}"] = task
            
            self.logger.info(
                f"Subscribed to frames on channel {channel}",
                extra={"channel": channel}
            )
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"channel": channel},
                "subscribe_frame"
            )
            raise
    
    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel."""
        try:
            # Cancel message subscription
            msg_key = f"msg:{channel}"
            if msg_key in self.subscriptions:
                self.subscriptions[msg_key].cancel()
                try:
                    await self.subscriptions[msg_key]
                except asyncio.CancelledError:
                    pass
                del self.subscriptions[msg_key]
                await self.pubsub.unsubscribe(msg_key)
            
            # Cancel frame subscription
            frame_key = f"frame:{channel}"
            if frame_key in self.subscriptions:
                self.subscriptions[frame_key].cancel()
                try:
                    await self.subscriptions[frame_key]
                except asyncio.CancelledError:
                    pass
                del self.subscriptions[frame_key]
                await self.pubsub.unsubscribe(frame_key)
            
            self.logger.info(f"Unsubscribed from channel {channel}")
            
        except Exception as e:
            log_error_with_context(self.logger, e, {"channel": channel}, "unsubscribe")
    
    async def get_subscribers(self, channel: str) -> int:
        """Get number of subscribers for a channel."""
        try:
            # Check both message and frame channels
            msg_subs = await self.redis_client.pubsub_numsub(f"msg:{channel}")
            frame_subs = await self.redis_client.pubsub_numsub(f"frame:{channel}")
            
            total_subs = msg_subs[f"msg:{channel}"] + frame_subs[f"frame:{channel}"]
            return total_subs
            
        except Exception as e:
            log_error_with_context(self.logger, e, {"channel": channel}, "get_subscribers")
            return 0
    
    async def health_check(self) -> bool:
        """Check if Redis is healthy."""
        try:
            if not self._connected or not self.redis_client:
                return False
                
            await self.redis_client.ping()
            return True
            
        except Exception:
            return False
