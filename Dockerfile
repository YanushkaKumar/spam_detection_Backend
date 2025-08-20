FROM python:3.9-slim

WORKDIR /app
#cheack the code is working or not 
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libnlopt-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Debug: List the contents to verify models directory is copied
RUN echo "=== Container contents ===" && \
    ls -la /app && \
    echo "=== Models directory ===" && \
    ls -la /app/models/ || echo "Models directory not found!" && \
    echo "=== Python files ===" && \
    ls -la /app/*.py
    
RUN mkdir -p /app/logs

    
# Set proper permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "Api.py"]