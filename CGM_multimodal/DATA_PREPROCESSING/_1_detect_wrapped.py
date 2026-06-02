import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ---------------------------
# CONFIG
# ---------------------------

# Few-shot: put ~2–4 example image paths per class here (absolute or relative to CSV dir)
FEWSHOT_WRAPPED = [
    r"..\..\CGMacros\CGMacros-001\photos\00000032-PHOTO-2020-5-4-12-59-0.jpg",
    r"..\..\CGMacros\CGMacros-001\photos\00000049-PHOTO-2020-5-6-12-25-0.jpg"
]
FEWSHOT_VISIBLE = [
    #r"..\CGMacros-002\photos\00000057-PHOTO-2019-11-21-13-49-0.jpg",
    r"..\..\CGMacros\CGMacros-001\photos\00000016-PHOTO-2020-5-2-19-52-0.jpg",
    r"..\..\CGMacros\CGMacros-001\photos\00000021-PHOTO-2020-5-3-14-51-0.jpg"
]

# Prompt ensemble (text-only)
WRAPPED_PROMPTS = [
    # "a photo of food wrapped in aluminum foil, contents not visible",
    # "a wrapped burrito or sandwich in foil, food not visible",
    # "closed takeaway package or wrapped meal, contents hidden",
    "a photo of food wrapped in foil or paper where the food is not visible",
]
VISIBLE_PROMPTS = [
    # "a photo of visible food on a plate, food clearly visible",
    # "a meal in an open container where the food is clearly visible",
    # "close-up of cooked food with texture visible (rice, pasta, salad, meat)",
    "a photo of visible food on a plate or in an open container",
]

# Blend weights: alpha=0 -> only prompts, alpha=1 -> only few-shot prototypes
ALPHA = 0.5

# Decision controls
THRESHOLD = 0.9   # increase to reduce false positives
MARGIN = 0.1      # require confidence gap between wrapped vs visible

def image_features(processor, model, image_paths, device, batch_size=32):
    feats = []
    valid = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        imgs = []
        ok = []
        for p in batch:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                ok.append(True)
            except Exception:
                # placeholder so batching doesn't break
                imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                ok.append(False)
        inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)  # [B, D]
            emb = emb / emb.norm(dim=1, keepdim=True)
        feats.append(emb.cpu().numpy())
        valid.extend(ok)
    return np.vstack(feats), np.array(valid, dtype=bool)

def text_features(processor, model, prompts, device):
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)  # [P, D]
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb.cpu().numpy()

def mean_vec(x):
    v = x.mean(axis=0, keepdims=True)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v

def precompute_prototypes(processor, model, device):
    # ---------------------------
    # PRECOMPUTE TEXT PROTOTYPES (prompt ensemble)
    # ---------------------------
    w_text = mean_vec(text_features(processor, model, WRAPPED_PROMPTS, device=device))  # [1, D]
    v_text = mean_vec(text_features(processor, model, VISIBLE_PROMPTS, device=device))  # [1, D]
    w_img, v_img = None, None

    # ---------------------------
    # PRECOMPUTE FEW-SHOT IMAGE PROTOTYPES (optional)
    # ---------------------------
    use_fewshot = (len(FEWSHOT_WRAPPED) > 0 and len(FEWSHOT_VISIBLE) > 0)

    if use_fewshot:
        w_feats, w_ok = image_features(processor, model, FEWSHOT_WRAPPED, batch_size=16, device=device)
        v_feats, v_ok = image_features(processor, model, FEWSHOT_VISIBLE, batch_size=16, device=device)

        if not w_ok.all():
            raise RuntimeError("One of the WRAPPED few-shot example images couldn't be opened. Check paths.")
        if not v_ok.all():
            raise RuntimeError("One of the VISIBLE few-shot example images couldn't be opened. Check paths.")

        w_img = mean_vec(w_feats)
        v_img = mean_vec(v_feats)
    # else:
    #     # if no few-shot, fall back to text only
    #     ALPHA = 0.0
    return w_text, v_text, w_img, v_img, use_fewshot

def score_imgs(X, ok, w_text, v_text, w_img, v_img, use_fewshot, model):
    # similarities (cosine, since normalized)
    logit_w_text = (X @ w_text.T).squeeze(1)
    logit_v_text = (X @ v_text.T).squeeze(1)

    if use_fewshot:
        logit_w_img = (X @ w_img.T).squeeze(1)
        logit_v_img = (X @ v_img.T).squeeze(1)
    else:
        logit_w_img = np.zeros_like(logit_w_text)
        logit_v_img = np.zeros_like(logit_v_text)

    # blended logits
    logit_w = ALPHA * logit_w_img + (1 - ALPHA) * logit_w_text
    logit_v = ALPHA * logit_v_img + (1 - ALPHA) * logit_v_text

    # CLIP temperature / scale (important!)
    scale = model.logit_scale.exp().item()

    logit_w = logit_w * scale
    logit_v = logit_v * scale

    # softmax to probability of wrapped
    m = np.maximum(logit_w, logit_v)
    p_wrapped = np.exp(logit_w - m) / (np.exp(logit_w - m) + np.exp(logit_v - m))

    # decision with threshold + margin; invalid images -> mark as wrapped
    conf_gap = (logit_w - logit_v) / scale
    is_wrapped = (p_wrapped >= THRESHOLD) & (conf_gap >= MARGIN)
    is_wrapped = np.where(ok, is_wrapped, True)

    return p_wrapped, logit_w, logit_v, conf_gap, is_wrapped


def detect_wrapped(df, out_visible_path, out_wrapped_path, model_name="openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    df["full_image_path"] = [
        rf"..\..\CGMacros\CGMacros-{int(sub):03d}\{img_path}"
        for sub, img_path in zip(df["sub"], df["Image path"].astype(str))
    ]

    all_paths = df["full_image_path"].tolist()

    w_text, v_text, w_img, v_img, use_fewshot = precompute_prototypes(processor, model, device)

    X, ok = image_features(processor, model, all_paths, batch_size=64, device=device)
    if not ok.all():
        raise RuntimeError("One of the images couldn't be opened. Check paths.")

    p_wrapped, logit_w, logit_v, conf_gap, is_wrapped = score_imgs(X, ok, w_text, v_text, w_img, v_img, use_fewshot, model)

    df["PWrapped"] = p_wrapped
    df["LogitWrapped"] = logit_w
    df["LogitVisible"] = logit_v
    df["GapWrappedMinusVisible"] = conf_gap
    df["IsWrapped"] = is_wrapped

    df[df["IsWrapped"] == False].to_csv(out_visible_path, index=False)
    df[df["IsWrapped"] == True].to_csv(out_wrapped_path, index=False)

    print(f"Total: {len(df)} | Visible: {(~df['IsWrapped']).sum()} | Wrapped/Other: {df['IsWrapped'].sum()}")
    print(f"Saved: {out_visible_path}, {out_wrapped_path}")

    return df[df["IsWrapped"] == False]

if __name__ == "__main__":
    CSV_PATH = '../splits/lunch_dinner/data_correct_lunch_dinner_gl_stats.csv'  # your CSV with "Image path"
    OUT_VISIBLE = "../splits/lunch_dinner/data_correct_lunch_dinner_visible_gl_stats.csv"
    OUT_WRAPPED = "../splits/lunch_dinner/data_correct_lunch_dinner_wrapped_gl_stats.csv"

    df = pd.read_csv(CSV_PATH)

    visible_df = detect_wrapped(df, OUT_VISIBLE, OUT_WRAPPED)