# Dockerfile for the Flask backend
# Uses a slim Python image to keep the image size manageable.
# The frontend is served separately via docker-compose as a Node.js container.

FROM python:3.11-slim

WORKDIR /app

# Install system packages needed by some Python deps (e.g. torch, matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache this layer.
# The layer only rebuilds when requirements.txt changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codebase
COPY . .

# Expose Flask port
EXPOSE 5000

# Start the unified backend
CMD ["python", "app.py"]
