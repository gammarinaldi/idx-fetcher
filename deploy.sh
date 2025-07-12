#!/bin/bash

# IDX Fetcher Scheduler Deployment Script
# This script sets up and deploys the scheduler on a Linux VPS

set -e

echo "🚀 Starting IDX Fetcher Scheduler Deployment..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from example..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "✅ Created .env file from env.example"
        echo "📝 Please edit .env file with your configuration before running the scheduler"
    else
        echo "❌ env.example not found. Please create a .env file manually."
        exit 1
    fi
fi

# Check if stock_list.csv exists
if [ ! -f stock_list.csv ]; then
    echo "⚠️  stock_list.csv not found. Please ensure you have a stock list CSV file."
    echo "📝 The file should contain a 'Kode' column with stock codes."
    exit 1
fi

# Build and start the containers
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting scheduler service..."
docker-compose up -d

# Check if the service is running
echo "⏳ Waiting for service to start..."
sleep 10

if docker-compose ps | grep -q "Up"; then
    echo "✅ Scheduler is running successfully!"
    echo ""
    echo "📊 Useful commands:"
    echo "  View logs: docker-compose logs -f"
    echo "  Stop service: docker-compose down"
    echo "  Restart service: docker-compose restart"
    echo "  View status: docker-compose ps"
    echo ""
    echo "📁 Data and logs are stored in:"
    echo "  - ./data/ (CSV files and results)"
    echo "  - ./logs/ (Scheduler logs)"
    echo ""
    echo "🕐 The scheduler will run fetch_daily_market_data.py every day at 21:00 UTC+7"
else
    echo "❌ Service failed to start. Check logs with: docker-compose logs"
    exit 1
fi 