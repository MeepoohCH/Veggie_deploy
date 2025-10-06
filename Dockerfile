# =========================
# Base Image (มี YOLO + PyTorch แล้ว)
# =========================
FROM ultralytics/ultralytics:latest

WORKDIR /app

# คัดลอกไฟล์ทั้งหมดเข้า container
COPY . .

# ติดตั้ง dependencies จาก requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# เปิดพอร์ต 5000 (ให้ API ใช้)
EXPOSE 5000

# =========================
# ใช้ gunicorn + uvicorn workers รัน FastAPI
# =========================
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:5000", "--timeout", "300"]
