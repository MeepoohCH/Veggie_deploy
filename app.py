import io, json, traceback
from pathlib import Path
from PIL import Image
import torch, timm
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths อยู่ใน project เดียวกับ container
LEAF_MODEL_PATH   = Path("leaf_model.pt")
TYPE_MODEL_PATH   = Path("type_model.pt")
LEAF_CLASSES_PATH = Path("leaf_classes.json")
TYPE_CLASSES_PATH = Path("type_classes.json")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _load_classes(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raise RuntimeError(f"Cannot find {path}")

def _build_model(name: str, num_classes: int):
    if "efficientnet" in name.lower():
        return timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=num_classes)
    else:
        return timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=num_classes)

def _safe_load_state_dict(model, ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state_dict, strict=False)
    return model

def _make_eval_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(size * 1.1)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

@torch.no_grad()
def _predict(model, img: Image.Image, tfm, device=DEVICE):
    x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = probs.max(dim=0)
    return int(idx.item()), float(conf.item()), probs.cpu().numpy().tolist()

# Load models & classes
LEAF_CLASSES = _load_classes(LEAF_CLASSES_PATH)
TYPE_CLASSES = _load_classes(TYPE_CLASSES_PATH)

leaf_model = _safe_load_state_dict(_build_model("mobilenetv3_large_100", len(LEAF_CLASSES)), LEAF_MODEL_PATH).to(DEVICE).eval()
type_model = _safe_load_state_dict(_build_model("tf_efficientnetv2_s", len(TYPE_CLASSES)), TYPE_MODEL_PATH).to(DEVICE).eval()

tf_leaf = _make_eval_transform(224)
tf_type  = _make_eval_transform(256)

# FastAPI
app = FastAPI(title="Leaf & Type Classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "leaf_classes": LEAF_CLASSES, "type_classes": TYPE_CLASSES}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes))

        leaf_idx, leaf_conf, _ = _predict(leaf_model, img, tf_leaf)
        leaf_label = LEAF_CLASSES[leaf_idx]

        type_idx, type_conf, type_probs = _predict(type_model, img, tf_type)
        type_label = TYPE_CLASSES[type_idx]

        return JSONResponse({
            "leaf": {"class": leaf_label, "confidence": leaf_conf},
            "type": {
                "class": type_label,
                "confidence": type_conf,
                "probs": {TYPE_CLASSES[i]: float(type_probs[i]) for i in range(len(TYPE_CLASSES))}
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)
