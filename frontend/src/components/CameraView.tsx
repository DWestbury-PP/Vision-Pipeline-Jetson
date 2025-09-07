import React, { useRef, useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Switch } from './ui/switch';
import { Camera, Square, RotateCcw } from 'lucide-react';

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

interface FrameMetadata {
  frame_id: number;
  timestamp: string;
  width: number;
  height: number;
  channels: number;
  fps: number;
  camera_id: string;
}

interface CameraViewProps {
  isConnected: boolean;
  frameData?: string;
  frameMetadata?: FrameMetadata;
  detectionResult?: DetectionResult;
  onToggleBoundingBoxes: (enabled: boolean) => void;
  onToggleMirrorMode: (enabled: boolean) => void;
  onToggleConfidence: (enabled: boolean) => void;
  showBoundingBoxes: boolean;
  mirrorMode: boolean;
  showConfidence: boolean;
}

export const CameraView: React.FC<CameraViewProps> = ({
  isConnected,
  frameData,
  frameMetadata,
  detectionResult,
  onToggleBoundingBoxes,
  onToggleMirrorMode,
  onToggleConfidence,
  showBoundingBoxes,
  mirrorMode,
  showConfidence,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 640, height: 480 });

  useEffect(() => {
    if (!frameData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Apply mirror mode transform
      if (mirrorMode) {
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-canvas.width, 0);
      }
      
      // Draw the frame
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      // Restore transform for drawing overlays
      if (mirrorMode) {
        ctx.restore();
      }
      
      // Draw bounding boxes if enabled
      if (showBoundingBoxes && detectionResult) {
        drawBoundingBoxes(ctx, detectionResult.bounding_boxes, canvas.width, canvas.height, mirrorMode);
      }
    };
    
    img.src = frameData;
  }, [frameData, showBoundingBoxes, mirrorMode, showConfidence, detectionResult]);

  const drawBoundingBoxes = (
    ctx: CanvasRenderingContext2D,
    boxes: BoundingBox[],
    canvasWidth: number,
    canvasHeight: number,
    mirrorMode: boolean
  ) => {
    if (!frameMetadata) return;

    const scaleX = canvasWidth / frameMetadata.width;
    const scaleY = canvasHeight / frameMetadata.height;

    boxes.forEach((box) => {
      let x = box.x1 * scaleX;
      let y = box.y1 * scaleY;
      let width = (box.x2 - box.x1) * scaleX;
      let height = (box.y2 - box.y1) * scaleY;
      
      // If mirror mode is enabled, flip the x coordinates
      if (mirrorMode) {
        // Mirror the bounding box horizontally
        const mirroredX1 = canvasWidth - (x + width);
        x = mirroredX1;
      }

      // Draw bounding box
      ctx.strokeStyle = '#10b981'; // emerald-500 - better for dark mode
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      // Draw label background
      const label = showConfidence 
        ? `${box.class_name} (${(box.confidence * 100).toFixed(1)}%)`
        : box.class_name;
        
      ctx.font = '14px Arial';
      const textMetrics = ctx.measureText(label);
      const textWidth = textMetrics.width;
      const textHeight = 20;

      ctx.fillStyle = 'rgba(16, 185, 129, 0.9)'; // emerald-500 with opacity
      ctx.fillRect(x, y - textHeight, textWidth + 8, textHeight);

      // Draw label text
      ctx.fillStyle = '#ffffff'; // white text for better contrast on dark background
      ctx.fillText(label, x + 4, y - 4);
    });
  };

  const handleCanvasResize = () => {
    if (canvasRef.current?.parentElement) {
      const container = canvasRef.current.parentElement;
      const maxWidth = container.clientWidth;
      const maxHeight = container.clientHeight - 100; // Leave space for controls
      
      // Maintain aspect ratio
      const aspectRatio = frameMetadata ? frameMetadata.width / frameMetadata.height : 16 / 9;
      let width = maxWidth;
      let height = width / aspectRatio;
      
      if (height > maxHeight) {
        height = maxHeight;
        width = height * aspectRatio;
      }
      
      setCanvasSize({ width, height });
    }
  };

  useEffect(() => {
    handleCanvasResize();
    window.addEventListener('resize', handleCanvasResize);
    return () => window.removeEventListener('resize', handleCanvasResize);
  }, [frameMetadata]);

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Camera className="h-5 w-5" />
          Camera View
          <div className={`ml-2 h-2 w-2 rounded-full ${
            isConnected ? 'bg-emerald-400' : 'bg-red-400'
          }`} />
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col h-full">
        {/* Camera Display */}
        <div className="flex-1 flex items-center justify-center bg-black rounded-lg overflow-hidden">
          {isConnected && frameData ? (
            <canvas
              ref={canvasRef}
              width={canvasSize.width}
              height={canvasSize.height}
              className="max-w-full max-h-full"
            />
          ) : (
            <div className="text-white text-center">
              <Camera className="h-16 w-16 mx-auto mb-4 opacity-50" />
              <p>{isConnected ? 'Waiting for video...' : 'Camera disconnected'}</p>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="mt-4 space-y-4">
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center space-x-2">
              <Switch
                checked={showBoundingBoxes}
                onCheckedChange={onToggleBoundingBoxes}
              />
              <label className="text-sm font-medium">
                Show Bounding Boxes
              </label>
            </div>
            
            <div className="flex items-center space-x-2">
              <Switch
                checked={showConfidence}
                onCheckedChange={onToggleConfidence}
              />
              <label className="text-sm font-medium">
                Show Confidence
              </label>
            </div>
            
            <div className="flex items-center space-x-2">
              <Switch
                checked={mirrorMode}
                onCheckedChange={onToggleMirrorMode}
              />
              <label className="text-sm font-medium">
                Mirror Mode
              </label>
            </div>
          </div>

          {/* Frame Info */}
          {frameMetadata && (
            <div className="text-xs text-muted-foreground grid grid-cols-2 md:grid-cols-4 gap-2">
              <span>Frame: #{frameMetadata.frame_id}</span>
              <span>Resolution: {frameMetadata.width}x{frameMetadata.height}</span>
              <span>FPS: {frameMetadata.fps.toFixed(1)}</span>
              <span>
                Detections: {detectionResult?.bounding_boxes.length || 0}
              </span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
