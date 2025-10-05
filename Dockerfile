# Base image
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# อัปเดตและติดตั้ง dependencies ของระบบ
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        && rm -rf /var/lib/apt/lists/*

# คัดลอก requirements
COPY requirements.txt .

# ติดตั้ง Python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# คัดลอกโค้ด
COPY . .

# Expose port
EXPOSE 5000

# Run app ด้วย gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
