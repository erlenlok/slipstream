#!/bin/bash
set -e

SERVICE_FILE="brawler.service"
LOCAL_PATH="/home/ubuntu/slipstream/$SERVICE_FILE"
SYSTEM_PATH="/etc/systemd/system/$SERVICE_FILE"

echo "--- Brawler Deployment ---"
echo "Source: $LOCAL_PATH"
echo "Dest:   $SYSTEM_PATH"

if [ ! -f "$LOCAL_PATH" ]; then
    echo "Error: Local service file not found at $LOCAL_PATH"
    exit 1
fi

# 1. Update System File
echo "1. Copying service file to systemd directory..."
sudo cp "$LOCAL_PATH" "$SYSTEM_PATH"

# 2. Reload Daemon
echo "2. Reloading systemd daemon..."
sudo systemctl daemon-reload

# 3. Restart Service
echo "3. Restarting brawler service..."
sudo systemctl restart brawler

# 4. Verification
echo "4. Checking status..."
sudo systemctl status brawler --no-pager

echo "--- Deployment Successful ---"
