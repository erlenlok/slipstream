#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "🚀 Starting Brawler Analytics Database Setup..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed.${NC}"
    echo "Attempting to install Docker automatically..."
    
    # Update and install
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose-v2
    
    # Start service
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Add user to group
    sudo usermod -aG docker $USER
    
    echo -e "${GREEN}Docker installed successfully!${NC}"
    
    # We need to run the rest of the script with the new group membership
    # using 'sg' (execute command as different group ID)
    echo "Restarting setup with new permissions..."
    exec sg docker "$0"
fi

# Check if we have permission to run docker
if ! docker ps &> /dev/null; then
    echo "Configuring Docker permissions for current session..."
    if ! groups | grep -q docker; then
         sudo usermod -aG docker $USER
    fi
    # Re-run script with docker group
    exec sg docker "$0"
fi


echo "📦 Spinning up TimescaleDB container..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d
else
    docker compose up -d
fi

echo "⏳ Waiting for database to be ready..."
# Loop to check health
MAX_RETRIES=30
COUNT=0
while [ $COUNT -lt $MAX_RETRIES ]; do
    if docker ps | grep -q "slipstream_analytics_db"; then
        if docker exec slipstream_analytics_db pg_isready -U postgres > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Database is up and running!${NC}"
            
            echo -e "\n📊 Connection Details:"
            echo "   Host: localhost"
            echo "   Port: 5432"
            echo "   User: postgres"
            echo "   Pass: postgres"
            echo "   DB:   slipstream_analytics"
            echo "   Volume: slipstream_db_data (Persistent)"
            
            echo -e "\nTo stop the database: docker-compose down"
            echo "To see logs: docker-compose logs -f"
            exit 0
        fi
    fi
    echo -n "."
    sleep 1
    COUNT=$((COUNT+1))
done

echo -e "\n${RED}❌ Timeout waiting for database to start.${NC}"
docker logs slipstream_analytics_db
exit 1
