"""
Direct iAUC prediction with open-weight VLMs from CGMacros-style test.txt.

Supported backends:
  1) qwen:     Qwen/Qwen2.5-VL-7B-Instruct
  2) internvl: OpenGVLab/InternVL3-8B-Instruct or OpenGVLab/InternVL3_5-8B-Instruct

Example:
python vlm_direct_iauc.py \
  --backend qwen \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --txt ../cgmacros/lunch_sub_train_test_val/test.txt \
  --image_root ../cgmacros \
  --out CHECKPOINTS/vlm_qwen25vl_test_iauc.csv

python vlm_direct_iauc.py \
  --backend internvl \
  --model_id OpenGVLab/InternVL3-8B-Instruct \
  --txt ../cgmacros/lunch_sub_train_test_val/test.txt \
  --image_root ../cgmacros \
  --out CHECKPOINTS/vlm_internvl3_test_iauc.csv
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


# ---------------------------------------------------------------------
# Column mapping for your uploaded CGMacros-style test.txt
# ---------------------------------------------------------------------
# 0: image path
# 1: image id
# 2: true calories             -> do NOT include for direct VLM baseline
# 3: meal type / split label    -> optional, ignored here
# 4-7: true meal nutrition      -> do NOT include for direct VLM baseline
# 8-22: clinical variables
# 23-27: microbiome variables
# 28: true iAUC120
# 29: true AUC120

CLINICAL_COLS = {
    "Baseline Libre glucose": 8,
    "Age": 9,
    "Gender": 10,
    "BMI": 11,
    "A1c": 12,
    "HOMA": 13,
    "Insulin": 14,
    "Triglycerides": 15,
    "Cholesterol": 16,
    "HDL": 17,
    "Non-HDL": 18,
    "LDL": 19,
    "VLDL": 20,
    "CHO/HDL ratio": 21,
    "Fasting blood glucose": 22,
}

MICROBIOME_COLS = {
    "Lachnospiraceae": 23,
    "Streptococcaceae": 24,
    "Bacteroidaceae": 25,
    "Enterobacteriaceae": 26,
    "Prevotellaceae": 27,
}

TRUE_IAUC_COL = 28
TRUE_AUC_COL = 29


def resolve_image_path(path_value: str, image_root: str | None) -> Path:
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    if image_root is not None:
        return Path(image_root) / p
    return p


def row_to_prompt(row: pd.Series) -> str:
    clinical_lines = []
    for name, col in CLINICAL_COLS.items():
        clinical_lines.append(f"- {name}: {row.iloc[col]}")

    microbiome_lines = []
    for name, col in MICROBIOME_COLS.items():
        microbiome_lines.append(f"- {name}: {row.iloc[col]}")

    prompt = f"""
You are given a food image and structured participant information.
Your task is to predict the 2-hour postprandial glucose incremental area under the curve, iAUC120.

Clinical variables:
{chr(10).join(clinical_lines)}

Microbiome features:
{chr(10).join(microbiome_lines)}

Do not explain your reasoning. Return exactly one JSON object with this schema:
{{"predicted_iAUC120": <number>}}
""".strip()
    return prompt




def extract_number(text: str) -> float:
    """Extract predicted_iAUC120 from JSON if possible, otherwise first number."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if "predicted_iAUC120" in obj:
            return float(obj["predicted_iAUC120"])
    except Exception:
        pass

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match is None:
        return np.nan
    return float(match.group(0))


# ---------------------------------------------------------------------
# Qwen2.5-VL backend
# ---------------------------------------------------------------------

def load_qwen(model_id: str):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor #, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )

    # max_memory = {
    #     0: "20GiB",
    #     #1: "10GiB",
    # }

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        # max_memory=max_memory,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     attn_implementation="eager",
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )

    model.eval()
    print("Device map:", getattr(model, "hf_device_map", "no device map"))

    return model, processor

def predict_qwen(model, processor, image_path: Path, prompt: str) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")

    # inputs = processor.apply_chat_template(
    #     messages,
    #     tokenize=True,
    #     add_generation_prompt=True,
    #     return_dict=True,
    #     return_tensors="pt",
    # )
    #
    # inputs = inputs.to(model.device)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64, #32
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text



# ---------------------------------------------------------------------
# InternVL backend
# ---------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_internvl_image(image_file: Path, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_internvl(model_id: str, load_8bit: bool = False):
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
        device_map="auto",
        max_memory={0: "20GiB"}, #1: "10GiB"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
    )
    return model, tokenizer


def predict_internvl(model, tokenizer, image_path: Path, prompt: str, dtype=torch.bfloat16) -> str:
    pixel_values = load_internvl_image(image_path, max_num=12)
    pixel_values = pixel_values.to(device="cuda", dtype=dtype)

    question = "<image>\n" + prompt

    generation_config = dict(
        max_new_tokens=64,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, question, generation_config)

    return response

def extract_prediction(raw_response):
    if raw_response is None:
        return math.nan

    text = str(raw_response).strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    # First try: parse JSON object
    try:
        obj = json.loads(text)
        if "predicted_iAUC120" in obj:
            return float(obj["predicted_iAUC120"])
    except Exception:
        pass

    # Second try: extract value after the exact key
    match = re.search(
        r'"?predicted_iAUC120"?\s*:\s*(-?\d+(?:\.\d+)?)',
        text
    )
    if match:
        return float(match.group(1))

    # Last fallback: extract the last number, not the first one
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return float(numbers[-1])

    return math.nan


# ---------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------

def run(args):
    df = pd.read_csv(args.txt, sep=r"\s+", header=None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "qwen":
        model, processor_or_tokenizer = load_qwen(args.model_id)
        predict_fn = lambda img, prompt: predict_qwen(model, processor_or_tokenizer, img, prompt)
    elif args.backend == "internvl":
        model, processor_or_tokenizer = load_internvl(args.model_id, load_8bit=args.load_8bit)
        predict_fn = lambda img, prompt: predict_internvl(model, processor_or_tokenizer, img, prompt)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    rows = []
    for idx, row in df.iterrows():
        image_path = resolve_image_path(row.iloc[0], args.image_root)
        prompt = row_to_prompt(row)

        if not image_path.exists():
            raw_response = f"ERROR: image not found: {image_path}"
            pred = np.nan
        else:
            try:
                raw_response = predict_fn(image_path, prompt)
                # pred = extract_number(raw_response)
                pred = extract_prediction(raw_response)
            except Exception as e:
                raw_response = f"ERROR: {repr(e)}"
                pred = np.nan

        true_iauc = float(row.iloc[TRUE_IAUC_COL])

        image_path = str(row.iloc[0])
        subject_id = re.search(r"CGMacros-(\d+)", image_path).group(1)

        rows.append({
            "row_idx": idx,
            "image_path": str(row.iloc[0]),
            "subject_id": subject_id,
            "y_true": true_iauc,
            "y_pred": pred,
            "raw_response": raw_response,
        })

        # Save after every row so a cluster job does not lose all progress.
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"[{idx + 1}/{len(df)}] true={true_iauc:.3f}, pred={pred}, response={raw_response}")

    out = pd.DataFrame(rows)
    valid = out.dropna(subset=["y_pred"])
    if len(valid) > 1:
        mae = np.mean(np.abs(valid["y_pred"] - valid["y_true"]))
        rmse = np.sqrt(np.mean((valid["y_pred"] - valid["y_true"]) ** 2))
        corr = valid[["y_pred", "y_true"]].corr().iloc[0, 1]
        print(f"Valid predictions: {len(valid)}/{len(out)}")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"Pearson r: {corr:.3f}")
    else:
        print("Not enough valid predictions to compute metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["qwen", "internvl"], required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--txt", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--load_8bit", action="store_true", help="Only used for InternVL backend")
    args = parser.parse_args()
    run(args)
