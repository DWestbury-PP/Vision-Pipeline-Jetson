"""WebSocket handler for real-time communication with the frontend."""

import asyncio
import json
import base64
import io
from typing import Dict, Set, Optional, Any
import numpy as np
from PIL import Image
from datetime import datetime

from ..shared.opencv_patch import cv2, CV2_AVAILABLE
from fastapi import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat() + 'Z'
        return super().default(obj)

from ..message_bus.redis_bus import RedisMessageBus
from ..message_bus.base import MessageBusSubscriber, Channels
from ..shared.models import (
    FrameMetadata, DetectionMessage, VLMMessage, StatusMessage, 
    ChatRequestMessage, ChatResponseMessage, ChatMessage
)
from .models import (
    WSFrameUpdate, WSDetectionUpdate, WSStatusUpdate, WSChatResponse, WSError
)
from ..shared.logging_config import setup_logging, log_error_with_context
from ..shared.config import get_config


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_settings: Dict[str, Dict[str, Any]] = {}
        self.logger = setup_logging("websocket_manager")
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_settings[client_id] = {
            "send_frames": True,
            "send_detections": True,
            "send_status": True,
            "frame_quality": 85,
            "frame_max_size": (640, 480)
        }
        
        self.logger.info(
            f"Client {client_id} connected",
            extra={"client_id": client_id, "total_connections": len(self.active_connections)}
        )
    
    def disconnect(self, client_id: str) -> None:
        """Remove a WebSocket connection."""
        self.active_connections.pop(client_id, None)
        self.connection_settings.pop(client_id, None)
        
        self.logger.info(
            f"Client {client_id} disconnected",
            extra={"client_id": client_id, "total_connections": len(self.active_connections)}
        )
    
    async def send_personal_message(self, message: dict, client_id: str) -> None:
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(json.dumps(message, cls=DateTimeEncoder))
            except Exception as e:
                log_error_with_context(
                    self.logger,
                    e,
                    {"client_id": client_id, "message_type": message.get("message_type")},
                    "send_personal_message"
                )
                # Remove disconnected client
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict) -> None:
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return
        
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message, cls=DateTimeEncoder))
            except Exception as e:
                log_error_with_context(
                    self.logger,
                    e,
                    {"client_id": client_id, "message_type": message.get("message_type")},
                    "broadcast"
                )
                disconnected_clients.append(client_id)
        
        # Remove disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def broadcast_filtered(self, message: dict, filter_func) -> None:
        """Broadcast a message to clients that match a filter."""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                if filter_func(client_id, self.connection_settings.get(client_id, {})):
                    await websocket.send_text(json.dumps(message, cls=DateTimeEncoder))
            except Exception as e:
                log_error_with_context(
                    self.logger,
                    e,
                    {"client_id": client_id, "message_type": message.get("message_type")},
                    "broadcast_filtered"
                )
                disconnected_clients.append(client_id)
        
        # Remove disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def update_client_settings(self, client_id: str, settings: dict) -> None:
        """Update settings for a specific client."""
        if client_id in self.connection_settings:
            self.connection_settings[client_id].update(settings)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)


class WebSocketHandler:
    """Handles WebSocket communication and message bus integration."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging("websocket_handler")
        
        # Connection management
        self.manager = ConnectionManager()
        
        # Message bus
        self.message_bus = RedisMessageBus()
        self.subscriber = MessageBusSubscriber(self.message_bus)
        self.publisher = None  # Will be set up when needed
        
        # Service state
        self.is_running = False
        self.subscription_tasks: Dict[str, asyncio.Task] = {}
        
        # Frame processing
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_metadata: Optional[FrameMetadata] = None
        self.frame_buffer_size = 3
        self.frame_send_interval = 0.1  # seconds
        
    async def start(self) -> None:
        """Start the WebSocket handler."""
        try:
            self.logger.info("Starting WebSocket handler")
            
            # Initialize message bus
            await self.subscriber.start()
            
            # Subscribe to relevant channels
            await self._setup_subscriptions()
            
            self.is_running = True
            
            self.logger.info("WebSocket handler started successfully")
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="websocket_handler_start")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the WebSocket handler."""
        self.logger.info("Stopping WebSocket handler")
        self.is_running = False
        
        # Cancel subscription tasks
        for task in self.subscription_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.subscription_tasks.clear()
        
        # Stop message bus
        try:
            await self.subscriber.stop()
        except Exception as e:
            log_error_with_context(self.logger, e, operation="message_bus_stop")
        
        self.logger.info("WebSocket handler stopped")
    
    async def _setup_subscriptions(self) -> None:
        """Set up message bus subscriptions."""
        # Subscribe to camera frames
        await self.subscriber.subscribe_to_frames(
            Channels.CAMERA_FRAMES,
            self._handle_frame_update
        )
        
        # Subscribe to detection results
        await self.subscriber.subscribe_to_messages(
            Channels.YOLO_RESULTS,
            self._handle_yolo_result
        )
        
        await self.subscriber.subscribe_to_messages(
            Channels.VLM_RESULTS,
            self._handle_vlm_result
        )
        
        await self.subscriber.subscribe_to_messages(
            Channels.FUSION_RESULTS,
            self._handle_fusion_result
        )
        
        # Subscribe to status updates
        await self.subscriber.subscribe_to_messages(
            Channels.SYSTEM_STATUS,
            self._handle_status_update
        )
        
        # Subscribe to chat responses
        await self.subscriber.subscribe_to_messages(
            Channels.CHAT_RESPONSES,
            self._handle_chat_response
        )
    
    async def _handle_frame_update(self, frame: np.ndarray, metadata: FrameMetadata) -> None:
        """Handle camera frame updates."""
        try:
            self.logger.info(f"Handling frame update for frame {metadata.frame_id}")
            self.latest_frame = frame
            self.latest_frame_metadata = metadata
            
            # Convert frame to base64 for WebSocket transmission
            frame_url = await self._frame_to_data_url(frame)
            self.logger.info(f"Converted frame {metadata.frame_id} to data URL")
            
            # Create WebSocket message
            ws_message = WSFrameUpdate(
                frame_metadata=metadata,
                frame_data_url=frame_url
            )
            
            # Broadcast to clients that want frame updates
            self.logger.info(f"Broadcasting frame {metadata.frame_id} to WebSocket clients")
            # Use dict() for Pydantic 1.x compatibility
            message_dict = ws_message.dict() if hasattr(ws_message, 'dict') else ws_message.model_dump(mode='json')
            await self.manager.broadcast_filtered(
                message_dict,
                lambda client_id, settings: settings.get("send_frames", True)
            )
            self.logger.info(f"Frame {metadata.frame_id} broadcast complete")
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"frame_id": metadata.frame_id},
                "handle_frame_update"
            )
    
    async def _handle_yolo_result(self, message: DetectionMessage) -> None:
        """Handle YOLO detection results."""
        try:
            self.logger.info(f"Received YOLO detection for frame {message.frame_id} with {len(message.result.bounding_boxes)} objects")
            
            ws_message = WSDetectionUpdate(
                frame_id=message.frame_id,
                yolo_result=message.result
            )
            
            self.logger.info(f"Broadcasting YOLO detection for frame {message.frame_id}")
            # Use dict() for Pydantic 1.x compatibility
            message_dict = ws_message.dict() if hasattr(ws_message, 'dict') else ws_message.model_dump(mode='json')
            await self.manager.broadcast_filtered(
                message_dict,
                lambda client_id, settings: settings.get("send_detections", True)
            )
            self.logger.info(f"YOLO detection broadcast complete for frame {message.frame_id}")
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"frame_id": message.frame_id},
                "handle_yolo_result"
            )
    
    async def _handle_vlm_result(self, message: VLMMessage) -> None:
        """Handle VLM results."""
        try:
            ws_message = WSDetectionUpdate(
                frame_id=message.frame_id,
                vlm_result=message.result
            )
            
            await self.manager.broadcast_filtered(
                ws_message.dict() if hasattr(ws_message, 'dict') else ws_message.model_dump(mode='json'),
                lambda client_id, settings: settings.get("send_detections", True)
            )
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"frame_id": message.frame_id},
                "handle_vlm_result"
            )
    
    async def _handle_fusion_result(self, message) -> None:
        """Handle fusion results."""
        try:
            # Extract ProcessingPipelineResult from FusionResultMessage
            if hasattr(message, 'result'):
                # It's a FusionResultMessage with the result wrapped
                fusion_result = message.result
            else:
                # Fallback for direct ProcessingPipelineResult (shouldn't happen)
                fusion_result = message
            
            # Now safely access frame_metadata
            if not hasattr(fusion_result, 'frame_metadata'):
                self.logger.warning(f"Fusion result missing frame_metadata: {type(fusion_result)}")
                return
                
            ws_message = WSDetectionUpdate(
                frame_id=fusion_result.frame_metadata.frame_id,
                fused_result=fusion_result
            )
            
            await self.manager.broadcast_filtered(
                ws_message.dict() if hasattr(ws_message, 'dict') else ws_message.model_dump(mode='json'),
                lambda client_id, settings: settings.get("send_detections", True)
            )
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                operation="handle_fusion_result"
            )
    
    async def _handle_status_update(self, message: StatusMessage) -> None:
        """Handle system status updates."""
        try:
            ws_message = WSStatusUpdate(status=message.status)
            
            await self.manager.broadcast_filtered(
                ws_message.dict() if hasattr(ws_message, 'dict') else ws_message.model_dump(mode='json'),
                lambda client_id, settings: settings.get("send_status", True)
            )
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="handle_status_update")
    
    async def _handle_chat_response(self, message: ChatResponseMessage) -> None:
        """Handle chat responses."""
        try:
            self.logger.info(f"Received chat response: {message.chat_response.response[:100]}...")
            
            ws_message = WSChatResponse(chat_response=message.chat_response)
            
            # Use dict() for Pydantic 1.x compatibility
            message_dict = ws_message.dict() if hasattr(ws_message, 'dict') else ws_message.model_dump(mode='json')
            await self.manager.broadcast(message_dict)
            
            self.logger.info("Chat response broadcast to WebSocket clients")
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="handle_chat_response")
    
    async def _frame_to_data_url(self, frame: np.ndarray, quality: int = 85) -> str:
        """Convert frame to base64 data URL."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Resize if needed (for bandwidth optimization)
            max_size = (640, 480)
            if pil_image.size[0] > max_size[0] or pil_image.size[1] > max_size[1]:
                pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to JPEG bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            
            # Encode to base64
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{image_data}"
            
            return data_url
            
        except Exception as e:
            log_error_with_context(self.logger, e, operation="frame_to_data_url")
            return ""
    
    async def handle_client_message(self, websocket: WebSocket, client_id: str, message: dict) -> None:
        """Handle incoming messages from WebSocket clients."""
        try:
            message_type = message.get("message_type")
            
            if message_type == "settings_update":
                # Update client settings
                settings = message.get("settings", {})
                self.manager.update_client_settings(client_id, settings)
                
                await self.manager.send_personal_message(
                    {"message_type": "settings_updated", "settings": settings},
                    client_id
                )
            
            elif message_type == "chat_message":
                # Handle chat message
                await self._handle_client_chat(message.get("data", {}), client_id)
            
            elif message_type == "ping":
                # Respond to ping
                await self.manager.send_personal_message(
                    {"message_type": "pong", "timestamp": message.get("timestamp")},
                    client_id
                )
            
            else:
                error_msg = WSError(
                    error_code="unknown_message_type",
                    error_message=f"Unknown message type: {message_type}"
                )
                error_dict = error_msg.dict() if hasattr(error_msg, 'dict') else error_msg.model_dump(mode='json')
                await self.manager.send_personal_message(error_dict, client_id)
                
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"client_id": client_id, "message_type": message.get("message_type")},
                "handle_client_message"
            )
            
            error_msg = WSError(
                error_code="message_processing_error",
                error_message="Error processing message"
            )
            error_dict = error_msg.dict() if hasattr(error_msg, 'dict') else error_msg.model_dump(mode='json')
            await self.manager.send_personal_message(error_dict, client_id)
    
    async def _handle_client_chat(self, chat_data: dict, client_id: str) -> None:
        """Handle chat message from client."""
        try:
            from datetime import datetime
            
            # Create chat message
            chat_message = ChatMessage(
                message=chat_data.get("message", ""),
                timestamp=datetime.utcnow(),
                frame_id=self.latest_frame_metadata.frame_id if self.latest_frame_metadata else None
            )
            
            # Create chat request message
            request_message = ChatRequestMessage(
                chat_message=chat_message,
                source_service="websocket_api"
            )
            
            # Publish to message bus (need publisher for this)
            if not self.publisher:
                from ..message_bus.base import MessageBusPublisher
                self.publisher = MessageBusPublisher(self.message_bus)
                await self.publisher.start()
            
            await self.publisher.publish_message(
                Channels.CHAT_REQUESTS,
                request_message
            )
            
            self.logger.info(f"Published chat request with message: {chat_message.message[:50]}...")
            
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"client_id": client_id, "chat_data": chat_data},
                "handle_client_chat"
            )
    
    async def handle_connection(self, websocket: WebSocket, client_id: str) -> None:
        """Handle a WebSocket connection lifecycle."""
        await self.manager.connect(websocket, client_id)
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                await self.handle_client_message(websocket, client_id, message)
                
        except WebSocketDisconnect:
            self.manager.disconnect(client_id)
        except Exception as e:
            log_error_with_context(
                self.logger,
                e,
                {"client_id": client_id},
                "handle_connection"
            )
            self.manager.disconnect(client_id)
    
    def get_stats(self) -> dict:
        """Get WebSocket handler statistics."""
        return {
            "active_connections": self.manager.get_connection_count(),
            "is_running": self.is_running,
            "latest_frame_id": self.latest_frame_metadata.frame_id if self.latest_frame_metadata else None,
            "subscriptions": list(self.subscription_tasks.keys())
        }
