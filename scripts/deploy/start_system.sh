#!/bin/bash
# Start Density Engine System

set -e

echo "Starting Density Engine System..."

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Starting Redis server..."
    redis-server --appendonly yes --appendfsync everysec --daemonize yes
    sleep 2
fi

# Check Redis connectivity
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Error: Redis is not accessible"
    exit 1
fi

echo "Redis is running"

# Start Celery workers
echo "Starting Celery workers..."
celery -A density_engine.tasks.celery_app worker \
    --loglevel=info \
    --concurrency=4 \
    --max-tasks-per-child=100 \
    --queues=jobs,instances,cluster \
    --hostname=worker@%h \
    --detach

# Start Celery beat scheduler (for periodic tasks)
echo "Starting Celery beat scheduler..."
celery -A density_engine.tasks.celery_app beat \
    --loglevel=info \
    --detach

# Start cluster manager
echo "Starting cluster manager..."
python scripts/manage_cluster.py &

echo "Density Engine System started successfully!"
echo "Components running:"
echo "  - Redis server"
echo "  - Celery workers"
echo "  - Celery beat scheduler"
echo "  - Cluster manager"
echo ""
echo "To stop the system, run: ./scripts/deploy/stop_system.sh"
