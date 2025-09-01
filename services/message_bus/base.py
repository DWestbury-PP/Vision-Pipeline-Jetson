"""Abstract base classes for message bus implementations."""

from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Dict, List
import asyncio
import numpy as np
from ..shared.models import BusMessage, FrameMetadata


class MessageBus(ABC):
    """Abstract base class for message bus implementations."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the message bus."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the message bus."""
        pass
    
    @abstractmethod
    async def publish_message(self, channel: str, message: BusMessage) -> None:
        """Publish a message to a channel.
        
        Args:
            channel: Channel name
            message: Message to publish
        """
        pass
    
    @abstractmethod
    async def publish_frame(
        self, 
        channel: str, 
        frame_data: np.ndarray, 
        metadata: FrameMetadata
    ) -> None:
        """Publish frame data to a channel.
        
        Args:
            channel: Channel name
            frame_data: Image frame as numpy array
            metadata: Frame metadata
        """
        pass
    
    @abstractmethod
    async def subscribe_message(
        self, 
        channel: str, 
        callback: Callable[[BusMessage], None]
    ) -> None:
        """Subscribe to messages on a channel.
        
        Args:
            channel: Channel name
            callback: Callback function for received messages
        """
        pass
    
    @abstractmethod
    async def subscribe_frame(
        self,
        channel: str,
        callback: Callable[[np.ndarray, FrameMetadata], None]
    ) -> None:
        """Subscribe to frame data on a channel.
        
        Args:
            channel: Channel name
            callback: Callback function for received frames
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel.
        
        Args:
            channel: Channel name
        """
        pass
    
    @abstractmethod
    async def get_subscribers(self, channel: str) -> int:
        """Get number of subscribers for a channel.
        
        Args:
            channel: Channel name
            
        Returns:
            Number of subscribers
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the message bus is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass


class MessageBusSubscriber:
    """Helper class for managing message bus subscriptions."""
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.subscriptions: Dict[str, Any] = {}
        self._running = False
    
    async def start(self) -> None:
        """Start the subscriber."""
        if not self._running:
            await self.message_bus.connect()
            self._running = True
    
    async def stop(self) -> None:
        """Stop the subscriber and unsubscribe from all channels."""
        if self._running:
            for channel in list(self.subscriptions.keys()):
                await self.unsubscribe(channel)
            await self.message_bus.disconnect()
            self._running = False
    
    async def subscribe_to_messages(
        self,
        channel: str,
        callback: Callable[[BusMessage], None]
    ) -> None:
        """Subscribe to messages on a channel."""
        await self.message_bus.subscribe_message(channel, callback)
        self.subscriptions[channel] = "message"
    
    async def subscribe_to_frames(
        self,
        channel: str,
        callback: Callable[[np.ndarray, FrameMetadata], None]
    ) -> None:
        """Subscribe to frames on a channel."""
        await self.message_bus.subscribe_frame(channel, callback)
        self.subscriptions[channel] = "frame"
    
    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel."""
        await self.message_bus.unsubscribe(channel)
        self.subscriptions.pop(channel, None)


class MessageBusPublisher:
    """Helper class for managing message bus publishing."""
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self._connected = False
    
    async def start(self) -> None:
        """Start the publisher."""
        if not self._connected:
            await self.message_bus.connect()
            self._connected = True
    
    async def stop(self) -> None:
        """Stop the publisher."""
        if self._connected:
            await self.message_bus.disconnect()
            self._connected = False
    
    async def publish_message(self, channel: str, message: BusMessage) -> None:
        """Publish a message to a channel."""
        await self.message_bus.publish_message(channel, message)
    
    async def publish_frame(
        self,
        channel: str,
        frame_data: np.ndarray,
        metadata: FrameMetadata
    ) -> None:
        """Publish frame data to a channel."""
        await self.message_bus.publish_frame(channel, frame_data, metadata)


# Standard channel names
class Channels:
    """Standard channel names for the message bus."""
    
    # Frame channels
    CAMERA_FRAMES = "camera.frames"
    
    # Detection channels
    YOLO_RESULTS = "detection.yolo"
    VLM_RESULTS = "detection.vlm"
    FUSION_RESULTS = "detection.fusion"
    
    # Chat channels
    CHAT_REQUESTS = "chat.requests"
    CHAT_RESPONSES = "chat.responses"
    
    # Status channels
    SYSTEM_STATUS = "system.status"
    SERVICE_STATUS = "service.status"
    
    # Configuration channels
    CONFIGURATION = "system.configuration"
    
    # Control channels
    CAMERA_CONTROL = "control.camera"
    SERVICE_CONTROL = "control.service"
