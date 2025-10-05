# Base image Python 3.10 slim
FROM python:3.10-slim

# กำหนด working directory
WORKDIR /app

# ติดตั้ง dependencies สำหรับ Pillow และ Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# คัดลอก requirements และติดตั้ง
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดทั้งหมดเข้า container
COPY . .

# ตั้งค่า Gunicorn สำหรับรัน Flask
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
