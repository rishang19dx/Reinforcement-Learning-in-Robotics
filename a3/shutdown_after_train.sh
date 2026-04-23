#!/bin/bash
# Wait for train_all.sh to finish, then shutdown
while pgrep -f "train_all.sh" > /dev/null; do
    sleep 60
done
echo "Training finished at $(date). Shutting down..."
sudo shutdown -h +1 "SAC training finished. Auto-shutdown."

