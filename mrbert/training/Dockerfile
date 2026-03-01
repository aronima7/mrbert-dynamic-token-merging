FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install Python dependencies (pinned to match Modal image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full mrbert package
COPY .. .

ENV PYTHONPATH=/workspace