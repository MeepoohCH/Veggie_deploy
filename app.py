from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import torch
from torchvision import transforms
import traceback

app = Flask(__name__)
CORS(app)  # อนุญาต cross-origin requests
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # จำกัดขนาดไฟล์ 5MB

# ===============================
# โหลดโมเดล PyTorch (CPU)
# ===============================
import torchvision.models as models

try:
    # -------- Leaf model --------
    leaf_checkpoint = torch.load("leaf_model.pt", map_location="cpu")
    leaf_state_dict = leaf_checkpoint["model"]  # เอาเฉพาะ key "model"

    leaf_model = models.mobilenet_v3_small(pretrained=False)
    leaf_model.classifier[3] = torch.nn.Linear(576, 2)  # 2 classes: leaf/not_leaf
    leaf_model.load_state_dict(leaf_state_dict)
    leaf_model.eval()
    print("✅ Leaf model (MobileNetV3) loaded successfully on CPU.")

    # -------- Type model --------
    type_checkpoint = torch.load("type_model.pt", map_location="cpu")
    type_state_dict = type_checkpoint["model"]

    type_model = models.efficientnet_v2_s(pretrained=False)
    type_model.classifier[1] = torch.nn.Linear(1280, 4)  # 4 classes: basil, spinach, mint, unknown
    type_model.load_state_dict(type_state_dict)
    type_model.eval()
    print("✅ Type model (EfficientNetV2) loaded successfully on CPU.")

except Exception as e:
    print("❌ ERROR: Could not load models.")
    print(traceback.format_exc())
    leaf_model, type_model = None, None


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
            "details": "One or both PyTorch models failed to load."
        }), 500

    try:
        if "image" not in request.files:
            return jsonify({"error": "Missing 'image' in request.files"}), 400

        file = request.files["image"]
        img_bytes = file.read()
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
            is_leaf = bool(leaf_pred.item() == 1)

        leaf_predictions = {
            "class": "leaf" if is_leaf else "not_leaf",
            "confidence": float(leaf_conf.item())
        }

        if not is_leaf:
            return jsonify({
                "leafCheck": "no",
                "leafPredictions": leaf_predictions,
                "message": "This image is not a leaf.",
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
        best_class = type_classes[type_pred.item()] if type_pred.item() < len(type_classes) else "unknown"
        best_conf = float(type_conf.item())

        print(f"🟢 Leaf detected → type: {best_class} ({best_conf:.2f})")

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
            "error": "Prediction failed",
            "details": str(e),
            "trace": detailed_error
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
