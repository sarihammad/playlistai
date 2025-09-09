FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy core library and install in development mode
COPY core/ ./core/
RUN pip install -e ./core

# Copy API application
COPY api/ ./api/

# Expose port
EXPOSE 8088

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8088/healthz || exit 1

# Run the application
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8088"]

