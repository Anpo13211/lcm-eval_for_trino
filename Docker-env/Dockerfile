# Trino LCM Evaluation Environment
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements/requirements_docker.txt /app/requirements.txt

# Install Python dependencies (excluding DGL for now)
RUN pip install --no-cache-dir -r requirements.txt

# Install DGL (CPU version for ARM64 compatibility)
RUN pip install dgl -f https://data.dgl.ai/wheels/repo.html

# Copy source code
COPY . /app/

# Set Python path
ENV PYTHONPATH=/app/src

# Default command
CMD ["python", "src/trino_lcm/scripts/test_docker_status.py"]
