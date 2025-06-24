FROM python:3.11-slim

ENV \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

WORKDIR /app

# Install ffmpeg + curl + git (if you need it)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      curl && \
    rm -rf /var/lib/apt/lists/*

# Copy & install only the packages your app actually uses
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Run under Gunicorn (or however you start your app)
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:5050", "app:app"]
