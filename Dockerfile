# Base image
FROM python:3.12-slim

WORKDIR /app

# ติดตั้ง OS dependencies สำหรับ Pillow และ Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# คัดลอกไฟล์ requirements
COPY requirements.txt .

# ติดตั้ง Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /tmp/pip-*

# คัดลอกไฟล์โปรเจกต์ทั้งหมด
COPY . .

# รันแอปด้วย Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
