# Use the slim Python image
FROM python:3.11-slim

# 1) Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      curl \
    && rm -rf /var/lib/apt/lists/*

# 2) Set working directory
WORKDIR /app

# 3) Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy application code and startup script
COPY . .

# 5) Expose the port (Render injects $PORT)
EXPOSE $PORT

# 6) Run the startup script at container launch
CMD ["./start.sh"]
