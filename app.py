from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import torch
from torchvision import transforms
import timm
import os
import traceback

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Transform สำหรับ inference (เหมือนตอนเทรน)
# ===============================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

def eval_transform(img):
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])(img)

# ===============================
# โหลดโมเดลแบบ safe
# ===============================
def load_leaf_model(path="leaf_model.pt"):
    full_path = os.path.join(os.getcwd(), path)
    try:
        checkpoint = torch.load(full_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=2)
        model.load_state_dict(state_dict)
        model.eval().to(DEVICE)
        print(f"✅ Leaf model loaded from {full_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load leaf model from {full_path}: {e}")
        traceback.print_exc()
        return None

def load_type_model(path="type_model.pt"):
    full_path = os.path.join(os.getcwd(), path)
    try:
        checkpoint = torch.load(full_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=3)
        model.load_state_dict(state_dict)
        model.eval().to(DEVICE)
        print(f"✅ Type model loaded from {full_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load type model from {full_path}: {e}")
        traceback.print_exc()
        return None

# ===============================
# โหลดโมเดลตอนเริ่มเซิร์ฟเวอร์
# ===============================
leaf_model = load_leaf_model()
type_model = load_type_model()

if leaf_model is None or type_model is None:
    raise RuntimeError("One or both PyTorch models failed to load. Check leaf_model.pt/type_model.pt")

# ===============================
# Predict route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Missing 'image' in request.files"}), 400

        file = request.files["image"]
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # -----------------
        # Leaf model
        leaf_tensor = eval_transform(img).unsqueeze(0).to(DEVICE)
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
                "message": "This image is not classified as a leaf.",
                "filename": file.filename,
                "image_size": img.size
            })

        # -----------------
        # Type model
        type_tensor = eval_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            type_out = type_model(type_tensor)
            type_probs = torch.nn.functional.softmax(type_out, dim=1)
            type_conf, type_pred = torch.max(type_probs, dim=1)

        type_classes = ["basil", "spinach", "mint"]
        best_class = type_classes[type_pred.item()]
        best_conf = float(type_conf.item())

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
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

# ===============================
# Health check
# ===============================
@app.route("/health")
def health():
    if leaf_model and type_model:
        return {"status": "ok", "message": "All models loaded successfully."}, 200
    return {"status": "model load failed", "message": "One or more models not initialized."}, 500

# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
