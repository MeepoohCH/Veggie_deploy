from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import torch
import timm
from torchvision import transforms
import traceback

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 5*1024*1024

# ===============================
# โมเดล timm
# ===============================
def build_gate_model(num_classes=2):
    return timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=num_classes)

def build_leaf_model(num_classes=3):
    return timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=num_classes)

def load_model(model_func, path):
    try:
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model = model_func()
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Model loaded from {path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
        traceback.print_exc()
        return None

leaf_model = load_model(lambda: build_gate_model(num_classes=2), "leaf_model.pt")
type_model = load_model(lambda: build_leaf_model(num_classes=3), "type_model.pt")

if leaf_model is None or type_model is None:
    raise RuntimeError("One or both models failed to load")

# ===============================
# Preprocess
# ===============================
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ===============================
# Flask routes
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error":"Missing 'image' in request.files"}),400
        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        leaf_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            leaf_out = leaf_model(leaf_tensor)
            leaf_probs = torch.nn.functional.softmax(leaf_out, dim=1)
            leaf_conf, leaf_pred = torch.max(leaf_probs, dim=1)
            is_leaf = bool(leaf_pred.item() == 1)
        leaf_result = {"class":"leaf" if is_leaf else "not_leaf","confidence":float(leaf_conf.item())}
        if not is_leaf:
            return jsonify({"leafCheck":"no","leafPredictions":leaf_result,"filename":file.filename,"image_size":img.size})
        type_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            type_out = type_model(type_tensor)
            type_probs = torch.nn.functional.softmax(type_out, dim=1)
            type_conf,type_pred = torch.max(type_probs,dim=1)
        type_classes = ["class1","class2","class3"]
        best_class = type_classes[type_pred.item()]
        return jsonify({
            "leafCheck":"yes",
            "leafPredictions":leaf_result,
            "bestPrediction":{"class":best_class,"confidence":float(type_conf.item())},
            "filename":file.filename,
            "image_size":img.size
        })
    except Exception as e:
        return jsonify({"error":"Prediction failed","details":str(e)}),500

@app.route("/health")
def health():
    return {"status":"ok","message":"All models loaded successfully."},200

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
