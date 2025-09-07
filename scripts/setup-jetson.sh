#!/bin/bash
# Jetson Setup Script for Vision Pipeline
# Configures NVIDIA Jetson for optimal performance and Docker GPU access

set -e

echo "🚀 Setting up NVIDIA Jetson for Vision Pipeline..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Jetson
check_jetson() {
    if [ ! -f /proc/device-tree/model ]; then
        print_error "Not running on a Jetson device!"
        exit 1
    fi
    
    MODEL=$(cat /proc/device-tree/model)
    print_status "Detected Jetson model: $MODEL"
}

# Set maximum performance mode
set_performance_mode() {
    print_status "Setting maximum performance mode..."
    
    # Enable maximum clocks
    if command -v jetson_clocks &> /dev/null; then
        sudo jetson_clocks
        print_status "Jetson clocks enabled"
    else
        print_warning "jetson_clocks not found, skipping clock optimization"
    fi
    
    # Set power mode to maximum (if nvpmodel exists)
    if command -v nvpmodel &> /dev/null; then
        sudo nvpmodel -m 0 2>/dev/null || print_warning "Could not set power mode to maximum"
        print_status "Power mode set to maximum"
    else
        print_warning "nvpmodel not found, skipping power mode optimization"
    fi
}

# Configure swap space (especially important for Jetson Nano)
configure_swap() {
    MODEL=$(cat /proc/device-tree/model)
    
    if [[ $MODEL == *"Nano"* ]]; then
        print_status "Configuring swap space for Jetson Nano..."
        
        # Check if swap already exists
        if swapon --show | grep -q "/swapfile"; then
            print_status "Swap file already configured"
            return
        fi
        
        # Create 4GB swap file
        sudo fallocate -l 4G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        
        # Add to fstab for persistence
        if ! grep -q "/swapfile" /etc/fstab; then
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        fi
        
        print_status "4GB swap file created and enabled"
    else
        print_status "Non-Nano Jetson detected, skipping swap configuration"
    fi
}

# Install/configure NVIDIA Docker runtime
configure_docker() {
    print_status "Configuring Docker with NVIDIA runtime..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if NVIDIA Docker runtime is installed
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        print_status "Installing NVIDIA Docker runtime..."
        
        # Add NVIDIA Docker repository
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        
        # Install nvidia-docker2
        sudo apt-get update
        sudo apt-get install -y nvidia-docker2
        
        print_status "NVIDIA Docker runtime installed"
    fi
    
    # Configure Docker daemon
    DAEMON_CONFIG="/etc/docker/daemon.json"
    
    if [ ! -f "$DAEMON_CONFIG" ]; then
        print_status "Creating Docker daemon configuration..."
        sudo tee "$DAEMON_CONFIG" > /dev/null <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
    else
        print_status "Docker daemon configuration already exists"
    fi
    
    # Restart Docker service
    print_status "Restarting Docker service..."
    sudo systemctl restart docker
    
    # Verify NVIDIA runtime
    if docker run --runtime=nvidia --rm nvcr.io/nvidia/l4t-base:r35.2.1 nvidia-smi &>/dev/null; then
        print_status "NVIDIA Docker runtime verified successfully"
    else
        print_error "NVIDIA Docker runtime verification failed"
        exit 1
    fi
}

# Install additional system dependencies
install_dependencies() {
    print_status "Installing system dependencies..."
    
    sudo apt-get update
    sudo apt-get install -y \
        v4l-utils \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        python3-gi \
        python3-gi-cairo \
        gir1.2-gstreamer-1.0 \
        git \
        git-lfs \
        curl \
        wget
    
    print_status "System dependencies installed"
}

# Test camera functionality
test_camera() {
    print_status "Testing camera functionality..."
    
    # List available cameras
    if command -v v4l2-ctl &> /dev/null; then
        print_status "Available cameras:"
        v4l2-ctl --list-devices || print_warning "No V4L2 cameras detected"
    fi
    
    # Test CSI camera with GStreamer
    print_status "Testing CSI camera with GStreamer..."
    if timeout 5 gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! fakesink &>/dev/null; then
        print_status "CSI camera test successful"
    else
        print_warning "CSI camera test failed - camera may not be connected"
    fi
}

# Create environment file
create_env_file() {
    if [ ! -f ".env" ]; then
        print_status "Creating .env file from template..."
        cp env.example .env
        print_status ".env file created"
    else
        print_status ".env file already exists"
    fi
}

# Main setup function
main() {
    print_status "Starting Jetson setup for Vision Pipeline..."
    
    check_jetson
    set_performance_mode
    configure_swap
    install_dependencies
    configure_docker
    test_camera
    create_env_file
    
    print_status "✅ Jetson setup completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "  1. Review and customize .env file if needed"
    echo "  2. Run: docker-compose up --build"
    echo "  3. Access the application at http://localhost:3000"
    echo ""
    print_warning "Note: First build may take 30-60 minutes to download and compile dependencies"
}

# Run main function
main "$@"
