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

# Ensure lib directory exists and copy utils.ts explicitly if needed
RUN ls -la src/lib/utils.ts || (echo "utils.ts not found, creating it..." && \
    mkdir -p src/lib && \
    echo 'import { type ClassValue, clsx } from "clsx"' > src/lib/utils.ts && \
    echo 'import { twMerge } from "tailwind-merge"' >> src/lib/utils.ts && \
    echo '' >> src/lib/utils.ts && \
    echo 'export function cn(...inputs: ClassValue[]) {' >> src/lib/utils.ts && \
    echo '  return twMerge(clsx(inputs))' >> src/lib/utils.ts && \
    echo '}' >> src/lib/utils.ts)

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
