import React, { useState, useEffect } from 'react';
import { CameraView } from './components/CameraView';
import { ChatInterface } from './components/ChatInterface';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Button } from './components/ui/button';
import { Switch } from './components/ui/switch';
import { Activity, Settings, Wifi, WifiOff } from 'lucide-react';
import './App.css';

interface SystemStatus {
  camera_active: boolean;
  yolo_active: boolean;
  moondream_instances: number;
  message_bus_connected: boolean;
  frames_processed: number;
  average_fps: number;
}

interface FrameMetadata {
  frame_id: number;
  timestamp: string;
  width: number;
  height: number;
  channels: number;
  fps: number;
  camera_id: string;
}

interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  class_name: string;
  class_id?: number;
}

interface DetectionResult {
  bounding_boxes: BoundingBox[];
  processing_time_ms: number;
  model_name: string;
  timestamp: string;
}

interface ChatMessage {
  id: string;
  message: string;
  timestamp: string;
  sender: 'user' | 'assistant';
  processing_time_ms?: number;
}

function App() {
  // Connection state
  const [isConnected, setIsConnected] = useState(false);
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  
  // System state
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  
  // Camera state
  const [frameData, setFrameData] = useState<string | undefined>();
  const [frameMetadata, setFrameMetadata] = useState<FrameMetadata | undefined>();
  const [detectionResult, setDetectionResult] = useState<DetectionResult | undefined>();
  
  // UI settings
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);
  const [mirrorMode, setMirrorMode] = useState(false);
  const [yoloEnabled, setYoloEnabled] = useState(true);
  const [vlmEnabled, setVlmEnabled] = useState(true);
  
  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isChatProcessing, setIsChatProcessing] = useState(false);

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const wsUrl = `ws://localhost:8000/ws/${Date.now()}`;
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setWebsocket(ws);
        
        // Send initial settings
        ws.send(JSON.stringify({
          message_type: 'settings_update',
          settings: {
            send_frames: true,
            send_detections: true,
            send_status: true,
            frame_quality: 85
          }
        }));
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleWebSocketMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        setWebsocket(null);
        
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };
    
    connectWebSocket();
    
    return () => {
      websocket?.close();
    };
  }, []);

  const handleWebSocketMessage = (message: any) => {
    switch (message.message_type) {
      case 'frame_update':
        setFrameData(message.frame_data_url);
        setFrameMetadata(message.frame_metadata);
        break;
        
      case 'detection_update':
        if (message.yolo_result) {
          setDetectionResult(message.yolo_result);
        }
        break;
        
      case 'status_update':
        setSystemStatus(message.status);
        break;
        
      case 'chat_response':
        setIsChatProcessing(false);
        setChatMessages(prev => [...prev, {
          id: Date.now().toString(),
          message: message.chat_response.response,
          timestamp: message.chat_response.timestamp,
          sender: 'assistant',
          processing_time_ms: message.chat_response.processing_time_ms
        }]);
        break;
        
      case 'error':
        console.error('WebSocket error:', message);
        break;
    }
  };

  const handleSendChatMessage = (message: string) => {
    if (!websocket || !isConnected) return;
    
    // Add user message to chat
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      message,
      timestamp: new Date().toISOString(),
      sender: 'user'
    };
    setChatMessages(prev => [...prev, userMessage]);
    setIsChatProcessing(true);
    
    // Send to backend
    websocket.send(JSON.stringify({
      message_type: 'chat_message',
      data: {
        message,
        include_current_frame: true
      }
    }));
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="h-8 w-8 text-primary" />
              <h1 className="text-2xl font-bold">Moondream Vision Pipeline</h1>
            </div>
            
            <div className="flex items-center gap-4">
              {/* Connection Status */}
              <div className="flex items-center gap-2">
                {isConnected ? (
                  <Wifi className="h-5 w-5 text-green-500" />
                ) : (
                  <WifiOff className="h-5 w-5 text-red-500" />
                )}
                <span className="text-sm">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              
              {/* System Stats */}
              {systemStatus && (
                <div className="text-sm text-muted-foreground">
                  {systemStatus.frames_processed} frames | {systemStatus.average_fps.toFixed(1)} FPS
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-200px)]">
          {/* Camera View - Takes up 2/3 on large screens */}
          <div className="lg:col-span-2">
            <CameraView
              isConnected={isConnected}
              frameData={frameData}
              frameMetadata={frameMetadata}
              detectionResult={detectionResult}
              onToggleBoundingBoxes={setShowBoundingBoxes}
              onToggleMirrorMode={setMirrorMode}
              onToggleConfidence={setShowConfidence}
              showBoundingBoxes={showBoundingBoxes}
              mirrorMode={mirrorMode}
              showConfidence={showConfidence}
            />
          </div>
          
          {/* Right Panel */}
          <div className="space-y-6">
            {/* System Controls */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  System Controls
                </CardTitle>
                <CardDescription>
                  Configure detection services
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">YOLO Detection</label>
                  <Switch
                    checked={yoloEnabled}
                    onCheckedChange={setYoloEnabled}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">VLM Processing</label>
                  <Switch
                    checked={vlmEnabled}
                    onCheckedChange={setVlmEnabled}
                  />
                </div>
                
                {systemStatus && (
                  <div className="pt-4 border-t space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span>Camera:</span>
                      <span className={systemStatus.camera_active ? 'text-green-600' : 'text-red-600'}>
                        {systemStatus.camera_active ? 'Active' : 'Inactive'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>YOLO:</span>
                      <span className={systemStatus.yolo_active ? 'text-green-600' : 'text-red-600'}>
                        {systemStatus.yolo_active ? 'Running' : 'Stopped'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Moondream:</span>
                      <span className={systemStatus.moondream_instances > 0 ? 'text-green-600' : 'text-red-600'}>
                        {systemStatus.moondream_instances} instances
                      </span>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
            
            {/* Chat Interface */}
            <div className="h-[400px]">
              <ChatInterface
                messages={chatMessages}
                onSendMessage={handleSendChatMessage}
                isProcessing={isChatProcessing}
                isConnected={isConnected}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;