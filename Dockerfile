FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by OpenCV & face-recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libsm6 \
    libxrandr2 \
    libxi6 \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
