# app.py
import io
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import timm
from torchvision import transforms, datasets

# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path ของโมเดล
GATE_PT = Path("runs/cls_gate_v2/final_gate_v2/best.pt")
LEAF_PT = Path("runs/cls_leaf3_v2/final_leaf3_v2/best.pt")


GATE_RUN_DIR = GATE_PT.parent
LEAF_RUN_DIR = LEAF_PT.parent

# =========================
# โหลดชื่อคลาส
# =========================
def load_class_names(run_dir: Path):
    cj = run_dir / "classes.json"
    if cj.exists():
        return json.loads(cj.read_text(encoding="utf-8"))
    raise RuntimeError(f"ไม่พบ classes.json ใน {run_dir}")

GATE_NAMES = load_class_names(GATE_RUN_DIR)
LEAF3_NAMES = load_class_names(LEAF_RUN_DIR)

# =========================
# สร้างโมเดล
# =========================
def build_gate_model(num_classes=len(GATE_NAMES)):
    return timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=num_classes)

def build_leaf_model(num_classes=len(LEAF3_NAMES)):
    return timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=num_classes)

# โหลด checkpoint
ck_gate = torch.load(GATE_PT, map_location=DEVICE)
ck_leaf = torch.load(LEAF_PT, map_location=DEVICE)

gate_model = build_gate_model()
leaf_model = build_leaf_model()

gate_model.load_state_dict(ck_gate["model"], strict=True)
leaf_model.load_state_dict(ck_leaf["model"], strict=True)

gate_model.to(DEVICE).eval()
leaf_model.to(DEVICE).eval()

# =========================
# Transform
# =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

GATE_SIZE = ck_gate.get("meta", {}).get("imgsz", 224)
LEAF_SIZE = ck_leaf.get("meta", {}).get("imgsz", 256)

def make_eval_transform(size):
    return transforms.Compose([
        transforms.Resize(int(size*1.1)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

tf_gate = make_eval_transform(GATE_SIZE)
tf_leaf = make_eval_transform(LEAF_SIZE)

# =========================
# Prediction functions
# =========================
@torch.no_grad()
def predict_probs(model, img_pil, tfm, device=DEVICE):
    x = tfm(img_pil.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    conf, idx = prob.max(dim=0)
    return int(idx.item()), float(conf.item()), prob.cpu().numpy()

def gate_decide(img_pil):
    idx, conf, _ = predict_probs(gate_model, img_pil, tf_gate)
    label = GATE_NAMES[idx]
    return label, conf, idx

def leaf_classify(img_pil):
    idx, conf, _ = predict_probs(leaf_model, img_pil, tf_leaf)
    return LEAF3_NAMES[idx], conf, idx

# =========================
# FastAPI
# =========================
app = FastAPI(title="Gate/Leaf Classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        gate_label, gate_conf, _ = gate_decide(img)

        if gate_label != "leaf" or gate_conf < 0.50:
            response_data = {
                "stage": "gate",
                "label": gate_label,
                "confidence": gate_conf
            }
        else:
            leaf_label, leaf_conf, _ = leaf_classify(img)
            response_data = {
                "stage": "leaf",
                "gate_label": gate_label,
                "gate_conf": gate_conf,
                "leaf_label": leaf_label,
                "leaf_conf": leaf_conf
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
