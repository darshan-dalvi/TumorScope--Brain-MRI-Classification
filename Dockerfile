# Backend Dockerfile for Brain Tumor Detection
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY requirements.txt pyproject.toml ./

# Install dependencies and the package
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app /home/appuser

# Set Python path to include the src directory
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/docs || exit 1

# Expose port
EXPOSE 8001

# Run the application
CMD ["uvicorn", "brain_tumor_detection.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]