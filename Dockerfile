# Use lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project files
COPY . .

# Expose port (if your app uses webcam/UI, this is still okay)
EXPOSE 8000

# Run the main file
CMD ["python", "main.py"]