#!/bin/bash
set -e

echo '[1/4] Checking NVIDIA CUDA environment...'
nvidia-smi

echo '[2/4] Downloading Cloudflare Tunnel (cloudflared)...'
if [ ! -f cloudflared ]; then
    wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
fi

echo '[3/4] Launching Cloudflare Tunnel for port 7860...'
./cloudflared tunnel --url http://127.0.0.1:7860 --logfile cloudflared.log 2>&1 &

SLIST_PID=$!
echo 'Waiting for Cloudflare Tunnel URL to generate...'
sleep 6
TUNNEL_URL=""
for i in {1..15}; do
    TUNNEL_URL=$(grep -o 'https://[-a-zA-Z0-9]*\.trycloudflare\.com' cloudflared.log | tail -n 1 || true)
    if [ ! -z "$TUNNEL_URL" ]; then
        break
    fi
    sleep 2
done

echo '=============================================================='
echo ' PUBLIC TEST URL (Open this on your Windows laptop):'
echo " $TUNNEL_URL"
echo '=============================================================='

echo '[4/4] Starting FastAPI WebRTC Pipecat Server on port 7860...'
export HOST="0.0.0.0"
export PORT="7860"
python server.py
