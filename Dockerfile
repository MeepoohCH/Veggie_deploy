# Base image: Python slim (เบากว่า ultralytics image)
FROM python:3.12-slim

WORKDIR /app
COPY . .

# ติดตั้ง dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# เปิดพอร์ต Flask
EXPOSE 5000

# รัน Flask ผ่าน gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
