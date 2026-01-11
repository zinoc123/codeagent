#!/bin/bash

# Exit on error
set -e

echo "--- Setting up AI Code Agent with Local LLM ---"

# 1. Check/Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment (venv)..."
    python3 -m venv venv
    echo "Installing dependencies..."
    ./venv/bin/pip install -r requirements.txt
else
    echo "Virtual environment found. Updating dependencies..."
    ./venv/bin/pip install -r requirements.txt
fi

# 2. Start Qdrant Docker Container
QDRANT_CONTAINER_NAME="qdrant_code_agent"

if [ "$(docker ps -q -f name=$QDRANT_CONTAINER_NAME)" ]; then
    echo "Qdrant container is already running."
else
    if [ "$(docker ps -aq -f name=$QDRANT_CONTAINER_NAME)" ]; then
        echo "Starting existing Qdrant container..."
        docker start $QDRANT_CONTAINER_NAME
    else
        echo "Starting new Qdrant container..."
        docker run -d -p 6333:6333 -p 6334:6334 \
            --name $QDRANT_CONTAINER_NAME \
            qdrant/qdrant:latest
    fi
fi

# 3. Start Ollama Docker Container (Local LLM)
OLLAMA_CONTAINER_NAME="ollama_local"

if [ "$(docker ps -q -f name=$OLLAMA_CONTAINER_NAME)" ]; then
    echo "Ollama container is already running."
else
    if [ "$(docker ps -aq -f name=$OLLAMA_CONTAINER_NAME)" ]; then
        echo "Starting existing Ollama container..."
        docker start $OLLAMA_CONTAINER_NAME
    else
        echo "Starting new Ollama container (with GPU)..."
        # Assuming NVIDIA Container Toolkit is set up for --gpus all
        docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 \
            --name $OLLAMA_CONTAINER_NAME \
            ollama/ollama
    fi
fi

echo "Waiting for services to be ready..."
sleep 5

# 4. Pull Local Models
echo "Pulling Qwen2.5-Coder:7b model (this may take a while)..."
docker exec $OLLAMA_CONTAINER_NAME ollama pull qwen2.5-coder:7b

echo "Pulling Nomic Embed Text model..."
docker exec $OLLAMA_CONTAINER_NAME ollama pull nomic-embed-text

# 5. Start FastAPI App
echo "Starting FastAPI Server..."
echo "Access the API at http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"

# Use exec to replace shell with python process
exec ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
