FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace

# Create a non-root user that matches a common host UID/GID (1000)
# to avoid permission issues with bind-mounted files
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --no-create-home appuser

USER appuser

ENV PYTHONPATH=/workspace

CMD ["/bin/bash"]
