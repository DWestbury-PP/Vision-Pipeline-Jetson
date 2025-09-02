#!/bin/bash

# Unified logging script for hybrid architecture
echo "🔍 Moondream Vision Pipeline - Unified Logs"
echo "============================================"

# Function to get container logs
get_container_logs() {
    local service=$1
    local lines=${2:-20}
    echo ""
    echo "📦 Container: $service"
    echo "----------------------------------------"
    if docker-compose ps | grep -q "$service"; then
        docker-compose logs --tail=$lines $service
    else
        echo "❌ Container $service not running"
    fi
}

# Function to get native service logs (if they exist)
get_native_logs() {
    local service=$1
    local logfile="/tmp/${service}_native.log"
    echo ""
    echo "🖥️  Native: $service"
    echo "----------------------------------------"
    if [ -f "$logfile" ]; then
        tail -20 "$logfile"
    else
        echo "❌ No log file found at $logfile"
        echo "💡 Native services log to stdout. Check running processes:"
        ps aux | grep -E "${service}_native" | grep -v grep || echo "   No ${service} process found"
    fi
}

# Function to show service status
show_status() {
    echo ""
    echo "📊 Service Status Overview"
    echo "============================================"
    
    echo ""
    echo "📦 Containerized Services:"
    docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    echo "🖥️  Native Services:"
    echo "Service         PID    Status"
    echo "--------------------------------"
    
    # Check Camera
    CAM_PID=$(ps aux | grep -E "camera_native\.py" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$CAM_PID" ]; then
        echo "Camera          $CAM_PID    ✅ Running"
    else
        echo "Camera          -      ❌ Not running"
    fi
    
    # Check YOLO
    YOLO_PID=$(ps aux | grep -E "yolo_native\.py" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$YOLO_PID" ]; then
        echo "YOLO11          $YOLO_PID    ✅ Running"
    else
        echo "YOLO11          -      ❌ Not running"
    fi
    
    # Check Moondream
    MOON_PID=$(ps aux | grep -E "moondream_native\.py" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$MOON_PID" ]; then
        echo "Moondream       $MOON_PID    ✅ Running"
    else
        echo "Moondream       -      ❌ Not running"
    fi
}

# Function to follow logs in real-time
follow_logs() {
    echo ""
    echo "📡 Following logs in real-time (Ctrl+C to stop)..."
    echo "============================================"
    
    # Start container log following in background
    docker-compose logs --follow --tail=5 redis api frontend fusion &
    DOCKER_PID=$!
    
    # Follow native service outputs if available
    echo "🖥️  Native services output to stdout/stderr"
    echo "   Check individual terminal windows or redirect to files"
    
    # Wait for user interrupt
    trap "kill $DOCKER_PID 2>/dev/null; echo ''; echo '✅ Log following stopped'; exit 0" INT
    wait $DOCKER_PID
}

# Main menu
case "${1:-status}" in
    "status"|"")
        show_status
        ;;
    "all")
        show_status
        get_container_logs "redis" 10
        get_container_logs "api" 15
        get_container_logs "frontend" 10
        get_container_logs "fusion" 10
        get_native_logs "yolo"
        get_native_logs "moondream"
        ;;
    "containers")
        get_container_logs "redis" 15
        get_container_logs "api" 20
        get_container_logs "frontend" 15
        get_container_logs "fusion" 15
        ;;
    "native")
        get_native_logs "yolo"
        get_native_logs "moondream"
        ;;
    "follow")
        follow_logs
        ;;
    "api")
        get_container_logs "api" 30
        ;;
    "frontend")
        get_container_logs "frontend" 30
        ;;
    "redis")
        get_container_logs "redis" 30
        ;;
    "yolo")
        get_native_logs "yolo"
        ;;
    "moondream")
        get_native_logs "moondream"
        ;;
    *)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  status      - Show service status overview (default)"
        echo "  all         - Show recent logs from all services"
        echo "  containers  - Show container logs only"
        echo "  native      - Show native service logs only"
        echo "  follow      - Follow logs in real-time"
        echo "  api         - Show API service logs"
        echo "  frontend    - Show frontend logs"
        echo "  redis       - Show Redis logs"
        echo "  yolo        - Show YOLO native service logs"
        echo "  moondream   - Show Moondream native service logs"
        echo ""
        echo "Examples:"
        echo "  ./scripts/logs-unified.sh status"
        echo "  ./scripts/logs-unified.sh all"
        echo "  ./scripts/logs-unified.sh follow"
        ;;
esac
