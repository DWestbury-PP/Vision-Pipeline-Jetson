# API service Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
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
python -m services.api.rest_api\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose API and WebSocket ports
EXPOSE 8000 8001

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the API service
ENTRYPOINT ["/app/entrypoint.sh"]
