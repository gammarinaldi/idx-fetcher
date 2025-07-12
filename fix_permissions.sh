#!/bin/bash

echo "🔧 Fixing permissions for IDX Fetcher..."

# Stop the container if running
echo "🛑 Stopping container..."
docker-compose down

# Create directories with proper permissions
echo "📁 Creating directories with proper permissions..."
mkdir -p data logs
chmod 755 data logs

# Try to set ownership (may fail if not root)
if [ "$EUID" -eq 0 ]; then
    echo "🔐 Setting ownership to user 1000..."
    chown -R 1000:1000 data logs
else
    echo "⚠️  Running as non-root, ownership may need manual adjustment"
    echo "💡 If you still have permission issues, run: sudo chown -R 1000:1000 data logs"
fi

# Rebuild and start
echo "🔨 Rebuilding container..."
docker-compose build --no-cache

echo "🚀 Starting container..."
docker-compose up -d

echo "✅ Permission fix complete!"
echo "📊 Check status with: docker-compose ps"
echo "📋 Check logs with: docker-compose logs -f" 