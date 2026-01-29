# Norse Docker Image
#
# Build:
#   docker build -t norse -f publish/Dockerfile --build-arg VERSION=1.0.0 .
#
# Run:
#   docker run -it --rm norse python -c "import norse; print(norse.__version__)"
#
# Run with mounted volume:
#   docker run -it --rm -v $(pwd):/workspace norse
#
# Run with GPU (requires nvidia-container-toolkit):
#   docker run -it --rm --gpus all norse
#
# Note: PyTorch includes CUDA support by default. GPU access requires
#       nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

FROM python:3.12-slim AS builder

# Version for setuptools-scm (required since .git is not copied)
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (includes CUDA support by default)
RUN pip install uv
RUN uv pip install --no-cache-dir --system torch torchvision

# Copy only what's needed for installation
COPY pyproject.toml README.md LICENSE ./
COPY norse/ ./norse/

# Install norse
RUN uv pip install --no-cache-dir --system .

# --- Production stage ---
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python"]
