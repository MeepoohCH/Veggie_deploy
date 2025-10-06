FROM python:3.12-slim

WORKDIR /app

# ติดตั้ง OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# อัพเกรด pip
RUN pip install --upgrade pip

# ติดตั้ง PyTorch + torchvision สำหรับ Python 3.12
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ติดตั้ง package ที่เหลือ
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
