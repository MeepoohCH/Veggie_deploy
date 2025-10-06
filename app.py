import io
import traceback
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# =========================
# สร้างแอป FastAPI
# =========================
app = FastAPI(title="YOLO Prediction API")

# อนุญาต CORS (สำคัญถ้ามี frontend เช่น Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # แนะนำให้กำหนดเฉพาะ origin ที่ต้องการใน production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# โหลดโมเดล YOLO
# =========================
try:
    model = YOLO("best.pt")
    print("✅ YOLO Model loaded successfully from best.pt")
except Exception as e:
    print(f"❌ ERROR: Could not load model 'best.pt'. Details: {e}")
    model = None


# =========================
# Endpoint /predict
# =========================
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model Initialization Error",
                "details": "YOLO model failed to load during server startup."
            }
        )

    try:
        # อ่านไฟล์จาก request
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes))
        print(f"📸 Received file: {image.filename}, size: {img.size}")

        # ใช้โมเดลทำนาย
        results = model(img)

        # ดึงผลลัพธ์ออกมา
        predictions = []
        if results and hasattr(results[0], 'boxes'):
            for r in results[0].boxes:
                predictions.append({
                    "class": model.names.get(int(r.cls[0]), "Unknown"),
                    "confidence": float(r.conf[0])
                })

        print(f"✅ Prediction successful. Found {len(predictions)} objects.")
        return JSONResponse(content=predictions)

    except Exception as e:
        detailed_error = traceback.format_exc()
        print("🚨 --- CRITICAL ERROR DURING PREDICTION ---")
        print(detailed_error)
        print("------------------------------------------")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction processing failed in FastAPI",
                "details": str(e),
                "trace": detailed_error
            }
        )


# =========================
# Health Check
# =========================
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}



