# Use a Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    lsof \
    && rm -rf /var/lib/apt/lists/*

# Download OPA binary
RUN curl -L -o /app/opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64 && \
    chmod +x /app/opa

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh /app/opa

# Expose ports (FastAPI: 8001, OPA: 8181)
EXPOSE 8001 8181

# Entrypoint script starts both processes
ENTRYPOINT ["/app/entrypoint.sh"]
