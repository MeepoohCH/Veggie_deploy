# Base image ที่มี PyTorch 2.3.0 + CUDA 12.1
FROM python:3.10-slim
WORKDIR /app

# ติดตั้ง dependencies เพิ่มเติม (ถ้ามีการประมวลผลภาพ)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# คัดลอก requirements และติดตั้ง Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์ทั้งหมดเข้า container
COPY . .

# รัน Flask ผ่าน Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
