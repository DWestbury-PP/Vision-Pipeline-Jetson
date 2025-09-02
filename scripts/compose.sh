#!/bin/bash

# Docker Compose wrapper that automatically disables BuildKit for local base images
# Usage: ./scripts/compose.sh up --build -d
#        ./scripts/compose.sh logs api --follow
#        ./scripts/compose.sh down

export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

# Forward all arguments to docker-compose
docker-compose "$@"
