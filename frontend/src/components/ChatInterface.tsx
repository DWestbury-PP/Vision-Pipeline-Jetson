import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Switch } from './ui/switch';
import { MessageCircle, Send, Loader2 } from 'lucide-react';

interface ChatMessage {
  id: string;
  message: string;
  timestamp: string;
  sender: 'user' | 'assistant';
  processing_time_ms?: number;
}

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  isProcessing: boolean;
  isConnected: boolean;
  vlmEnabled: boolean;
  onToggleVlm: (enabled: boolean) => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  isProcessing,
  isConnected,
  vlmEnabled,
  onToggleVlm,
}) => {
  const [inputMessage, setInputMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputMessage.trim() && !isProcessing && isConnected) {
      onSendMessage(inputMessage.trim());
      setInputMessage('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <Card className="h-full flex flex-col max-h-full">
      <CardHeader className="flex-shrink-0">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageCircle className="h-5 w-5" />
            VLM Chat
            <div className={`ml-2 h-2 w-2 rounded-full ${
              isConnected ? 'bg-emerald-400' : 'bg-red-400'
            }`} />
          </div>
          
          {/* VLM Toggle */}
          <Switch
            checked={vlmEnabled}
            onCheckedChange={onToggleVlm}
          />
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col flex-1 p-0 min-h-0">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
          {messages.length === 0 ? (
            <div className="text-center text-muted-foreground">
              <MessageCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>Start a conversation with the VLM</p>
              <p className="text-xs mt-2">
                Ask questions about what you see in the camera view
              </p>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${
                  msg.sender === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2 ${
                    msg.sender === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted text-muted-foreground'
                  }`}
                >
                  <p className="text-sm">{msg.message}</p>
                  <div className="flex items-center justify-between mt-1 text-xs opacity-70">
                    <span>{formatTimestamp(msg.timestamp)}</span>
                    {msg.processing_time_ms && (
                      <span>{msg.processing_time_ms.toFixed(0)}ms</span>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
          
          {isProcessing && (
            <div className="flex justify-start">
              <div className="bg-muted text-muted-foreground rounded-lg px-4 py-2 flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">VLM is thinking...</span>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t p-4 flex-shrink-0">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                isConnected
                  ? "Ask about what you see in the camera..."
                  : "Connect to start chatting..."
              }
              disabled={isProcessing || !isConnected || !vlmEnabled}
              className="flex-1 px-3 py-2 border border-input rounded-md bg-background text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            />
            <Button
              type="submit"
              size="sm"
              disabled={!inputMessage.trim() || isProcessing || !isConnected || !vlmEnabled}
            >
              {isProcessing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </form>
          
          {!isConnected && (
            <p className="text-xs text-muted-foreground mt-2">
              Waiting for connection to VLM service...
            </p>
          )}
          
          {!vlmEnabled && isConnected && (
            <p className="text-xs text-muted-foreground mt-2">
              VLM is disabled. Enable to start chatting...
            </p>
          )}
          
          {isConnected && vlmEnabled && (
            <p className="text-xs text-muted-foreground mt-2">
              Press Enter to send, Shift+Enter for new line
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
