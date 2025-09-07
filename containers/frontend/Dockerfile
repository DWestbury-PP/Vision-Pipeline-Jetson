# Frontend Dockerfile
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files first for better caching
COPY frontend/package.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY frontend/ ./

# Debug: List the directory structure to see what was copied
RUN echo "=== Debugging: Directory structure ===" && \
    find . -type f -name "*.ts" -o -name "*.tsx" | head -20 && \
    echo "=== Checking for utils.ts specifically ===" && \
    ls -la src/lib/ || echo "src/lib directory not found" && \
    echo "=== End debug info ==="

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app to nginx
COPY --from=0 /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY containers/frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
