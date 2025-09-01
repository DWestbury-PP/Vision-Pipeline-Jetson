# YOLO service Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY services/ ./services/
COPY env.example .env

# Create entrypoint script
RUN echo '#!/bin/bash\n\
cd /app\n\
# Download YOLO model if not exists\n\
python -c "from ultralytics import YOLO; YOLO('"'"'yolo11n.pt'"'"')"\n\
python -m services.cv_services.yolo_service\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='redis', port=6379); r.ping()" || exit 1

# Run the YOLO service
ENTRYPOINT ["/app/entrypoint.sh"]
