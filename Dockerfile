# Use an official NVIDIA CUDA runtime as the base image
FROM nvidia/cuda:12.4.1-runtime-ubuntu20.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    build-essential \
    wget \
    curl \
    # libgl1 \
    libgl1-mesa-glx \
    # libglib2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Create and activate a virtual environment
RUN python3.9 -m venv venv && \
    /bin/bash -c "source venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt"

# Expose the port your application will run on (e.g., FastAPI default port)
EXPOSE 8000

# Define the command to run your application
CMD ["/bin/bash", "-c", "source venv/bin/activate && python app.py"]
