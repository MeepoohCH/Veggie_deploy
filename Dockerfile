# =========================
# Stage 1: build environment
# =========================
FROM python:3.12-slim AS base

WORKDIR /app

# ติดตั้ง dependencies ที่จำเป็น
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์โปรเจกต์ทั้งหมด
COPY . .

# =========================
# Stage 2: runtime
# =========================
FROM python:3.12-slim
WORKDIR /app

# คัดลอกจาก stage ก่อนหน้า
COPY --from=base /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=base /usr/local/bin /usr/local/bin
COPY --from=base /app /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
