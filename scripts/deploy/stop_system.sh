#!/bin/bash
# Stop Density Engine System

set -e

echo "Stopping Density Engine System..."

# Stop cluster manager
echo "Stopping cluster manager..."
pkill -f "scripts/manage_cluster.py" || true

# Stop Celery beat scheduler
echo "Stopping Celery beat scheduler..."
celery -A density_engine.tasks.celery_app beat --stop || true

# Stop Celery workers
echo "Stopping Celery workers..."
celery -A density_engine.tasks.celery_app control shutdown || true

# Wait for workers to stop
sleep 5

# Stop Redis (optional - comment out if you want to keep Redis running)
# echo "Stopping Redis server..."
# redis-cli shutdown || true

echo "Density Engine System stopped successfully!"
