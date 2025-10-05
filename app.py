from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import torch
from torchvision import transforms, models
import traceback
import os

app = Flask(__name__)
CORS(app)  # อนุญาต cross-origin requests
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # จำกัดขนาดไฟล์ 5MB

# ===============================
# โหลดโมเดล PyTorch แบบ checkpoint
# ===============================
def load_leaf_model(path="leaf_model.pt"):
    """โหลด MobileNetV3 (Leaf/Not Leaf) และแมปไปที่ CPU"""
    full_path = os.path.join(os.getcwd(), path)
    
    try:
        # **สำคัญ:** ใช้ map_location="cpu" เพื่อให้รันได้โดยไม่มี GPU
        checkpoint = torch.load(full_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint) # รองรับทั้งแบบที่มี key 'model' และไม่มี

        model = models.mobilenet_v3_small(weights=None) # weights=None เพราะจะโหลดจาก state_dict
        model.classifier[-1] = torch.nn.Linear(1024, 2) # ปรับให้เข้ากับ MobileNetV3 small ที่ถูกต้อง (ตรวจสอบ Input features ของ Classifier ชั้นสุดท้าย)
        
        # สำหรับ MobileNetV3_small output feature ของก่อนชั้นสุดท้ายคือ 1024
        # ถ้าคุณใช้เวอร์ชันอื่นหรือปรับเอง อาจจะต้องเปลี่ยนเลข 1024
        
        # Note: โค้ดเดิมคือ model.classifier[3] = torch.nn.Linear(576, 2) ซึ่งอาจไม่ถูกต้องสำหรับ mobilenet_v3_small
        # ผมแก้เป็น model.classifier[-1] = torch.nn.Linear(1024, 2) 

        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Leaf model loaded from {full_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load leaf model from {full_path}:", e)
        # หากโหลดไม่ได้ (เช่น ไฟล์ไม่พบ) ให้พิมพ์ stack trace
        traceback.print_exc()
        return None

def load_type_model(path="type_model.pt"):
    """โหลด EfficientNetV2 (Type Classification) และแมปไปที่ CPU"""
    full_path = os.path.join(os.getcwd(), path)
    
    try:
        # **สำคัญ:** ใช้ map_location="cpu"
        checkpoint = torch.load(full_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint) # รองรับทั้งแบบที่มี key 'model' และไม่มี

        model = models.efficientnet_v2_s(weights=None)
        # EfficientNetV2-S มี 1280 features ก่อน Classifier ชั้นสุดท้าย
        model.classifier[1] = torch.nn.Linear(1280, 4)  # 4 classes: basil, spinach, mint, unknown

        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Type model loaded from {full_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load type model from {full_path}:", e)
        traceback.print_exc()
        return None

# โหลดโมเดล
# ตรวจสอบว่าโมเดลอยู่ที่ Working Directory ก่อนรัน Gunicorn
leaf_model = load_leaf_model()
type_model = load_type_model()

if leaf_model is None or type_model is None:
    # Raise error เพื่อให้ Gunicorn/Docker ทราบว่า initialization ล้มเหลว
    raise RuntimeError("One or both PyTorch models failed to load. Check model files (leaf_model.pt/type_model.pt) and paths.")

# ===============================
# Preprocess สำหรับ PyTorch โมเดล
# ===============================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

# ===============================
# Flask route /predict
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    if leaf_model is None or type_model is None:
        return jsonify({
            "error": "Model Initialization Error",
            "details": "One or both PyTorch models failed to load during startup."
        }), 500

    try:
        if "image" not in request.files:
            return jsonify({"error": "Missing 'image' in request.files"}), 400

        file = request.files["image"]
        img_bytes = file.read()
        # ใช้ Image.open(io.BytesIO(...)) เพื่อให้ PIL โหลดรูปภาพจากหน่วยความจำ
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        print(f"✅ Received image: {file.filename}, size: {img.size}")

        # ===============================
        # โมเดล 1: ตรวจใบ/ไม่ใช่ใบ
        # ===============================
        leaf_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            leaf_out = leaf_model(leaf_tensor)
            leaf_probs = torch.nn.functional.softmax(leaf_out, dim=1)
            leaf_conf, leaf_pred = torch.max(leaf_probs, dim=1)
            # 1 คือ "leaf", 0 คือ "not_leaf" (สมมติตามลำดับ class)
            is_leaf = bool(leaf_pred.item() == 1) 

        leaf_predictions = {
            "class": "leaf" if is_leaf else "not_leaf",
            "confidence": float(leaf_conf.item())
        }

        if not is_leaf:
            return jsonify({
                "leafCheck": "no",
                "leafPredictions": leaf_predictions,
                "message": "This image is not classified as a leaf.",
                "filename": file.filename,
                "image_size": img.size
            })

        # ===============================
        # โมเดล 2: จำแนกประเภทใบ
        # ===============================
        type_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            type_out = type_model(type_tensor)
            type_probs = torch.nn.functional.softmax(type_out, dim=1)
            type_conf, type_pred = torch.max(type_probs, dim=1)

        type_classes = ["basil", "spinach", "mint", "unknown"]
        best_class = type_classes[type_pred.item()] 
        best_conf = float(type_conf.item())

        print(f"🟢 Leaf detected → type: {best_class} (Conf: {best_conf:.4f})")

        return jsonify({
            "leafCheck": "yes",
            "leafPredictions": leaf_predictions,
            "bestPrediction": {
                "class": best_class,
                "confidence": best_conf
            },
            "filename": file.filename,
            "image_size": img.size
        })

    except Exception as e:
        detailed_error = traceback.format_exc()
        print("❌ ERROR DURING PREDICTION")
        print(detailed_error)
        return jsonify({
            "error": "Prediction failed due to internal error",
            "details": str(e)
        }), 500

# ===============================
# Health check (สำคัญสำหรับการ Deployment)
# ===============================
@app.route("/health")
def health():
    if leaf_model and type_model:
        return {"status": "ok", "message": "All models loaded successfully."}, 200
    return {"status": "model load failed", "message": "One or more PyTorch models are not initialized."}, 500

if __name__ == "__main__":
    # รันด้วย Flask's development server 
    app.run(host="0.0.0.0", port=5000, debug=True)
