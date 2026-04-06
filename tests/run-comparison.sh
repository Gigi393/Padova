#!/bin/bash
set -e

echo "=== Running Original Simulation ==="
python original_sim.py

echo ""
echo "=== Starting FastAPI Server ==="

# instead run docker-compose up --build -d 
cd ../ &&  docker-compose up --build -d && cd tests
# Start the server in the background
# python app.py &
# SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 5

echo ""
echo "=== Running Server-based Simulation and Comparison ==="
python client_script.py

# echo ""
# echo "=== Stopping Server ==="
# kill $SERVER_PID

cd ../ && docker-compose down

echo ""
echo "Comparison complete! Check simulation_comparison.png for results."
