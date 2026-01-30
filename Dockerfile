FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Runtime env
ENV PYTHONUNBUFFERED=1
ENV API_PORT=2147

EXPOSE 2147

CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "2147"]
