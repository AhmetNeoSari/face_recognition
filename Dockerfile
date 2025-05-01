# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-runtime-ubuntu20.04

# 1. Environment configuration
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CONDA_DIR=/opt/conda \
    PIP_CACHE_DIR=/root/.cache/pip \
    CONDA_PKGS_DIRS=/opt/conda/pkgs \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10

# 2. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 3. Miniconda installation
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# 4. Copy environment file FIRST
COPY environment.yaml .

# 5. Conda environment setup with cache
RUN --mount=type=cache,target=$CONDA_PKGS_DIRS \
    conda env create -f environment.yaml

# 6. Python package installation with extended timeouts
COPY requirements.txt .
RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    /bin/bash -c "source activate count_env && \
    pip install --retries 10 --default-timeout=1000 --cache-dir $PIP_CACHE_DIR -r requirements.txt"

# 7. Application copy
COPY . /app
WORKDIR /app

# 8. Runtime configuration
EXPOSE 8080
CMD ["/bin/bash", "-c", "source activate count_env && python app.py"]
