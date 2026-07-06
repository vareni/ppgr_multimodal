"""
Direct iAUC prediction with open-weight VLMs from CGMacros-style txt files.

Modes:
  image_clinical
      image + clinical variables + microbiome variables
      Does NOT include true nutrition/macronutrients.

  image_clinical_macros
      image + clinical variables + microbiome variables + true macronutrients

  clinical_macros
      clinical variables + microbiome variables + true macronutrients
      No image is passed to the VLM.

  clinical_only
      clinical variables + microbiome variables
      No image and no true macronutrients.

Examples:
python vlm_direct_iauc_modes.py \
  --backend internvl \
  --model_id OpenGVLab/InternVL3-2B-Instruct \
  --txt ../cgmacros/lunch_sub_train_test_val/test.txt \
  --image_root ../cgmacros \
  --input_mode clinical_macros \
  --out CHECKPOINTS/vlm_internvl3_2b_clinical_macros.csv

python vlm_direct_iauc_modes.py \
  --backend qwen \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --txt ../cgmacros/lunch_sub_train_test_val/test.txt \
  --image_root ../cgmacros \
  --input_mode image_clinical_macros \
  --out CHECKPOINTS/vlm_qwen25vl_7b_image_clinical_macros.csv
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
# Column mapping for CGMacros-style txt
# ---------------------------------------------------------------------
# 0: image path
# 1: image id
# 2: true calories
# 3: meal type / label, ignored here
# 4-7: true meal nutrition columns in your existing dataset format
# 8-22: clinical variables
# 23-27: microbiome variables
# 28: true iAUC120
# 29: true AUC120

TRUE_MACRO_COLS = {
    "Calories": 2,
    # These names follow the indexing used in your existing dataset/model code.
    # If your raw txt has a different order, adjust only these three lines.
    "Fat": 4,
    "Carbohydrates": 5,
    "Protein": 6,
}

CLINICAL_COLS = {
    "Baseline Libre glucose, mg/dL": 8,
    "Age, years": 9,
    "Gender, encoded": 10,
    "BMI, kg/m2": 11,
    "HbA1c, percent": 12,
    "HOMA-IR": 13,
    "Insulin": 14,
    "Triglycerides, mg/dL": 15,
    "Cholesterol, mg/dL": 16,
    "HDL, mg/dL": 17,
    "Non-HDL, mg/dL": 18,
    "LDL, mg/dL": 19,
    "VLDL, mg/dL": 20,
    "CHO/HDL ratio": 21,
    "Fasting blood glucose, mg/dL": 22,
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

IMAGE_MODES = {"image_clinical", "image_clinical_macros"}
MACRO_MODES = {"image_clinical_macros", "clinical_macros"}


def load_txt(path: str | Path) -> pd.DataFrame:
    # If your txt is comma-separated, change sep=r"\s+" to sep=","
    return pd.read_csv(path, sep=r"\s+", header=None)

def resolve_image_path(path_value: str, image_root: str | None) -> Path:
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    if image_root is not None:
        return Path(image_root) / p
    return p


def _format_lines(row: pd.Series, mapping: dict[str, int]) -> str:
    return "\n".join(f"- {name}: {row.iloc[col]}" for name, col in mapping.items())

def row_to_example(row: pd.Series, input_mode: str, i: int, use_microbiome: bool = True) -> str:
    use_macros = input_mode in MACRO_MODES

    micro_block = ""
    if use_microbiome:
        micro_block = f"""

Microbiome features:
{_format_lines(row, MICROBIOME_COLS)}
""".rstrip()

    macro_block = ""
    if use_macros:
        macro_block = f"""

True meal macronutrients:
{_format_lines(row, TRUE_MACRO_COLS)}
""".rstrip()

    return f"""
Example {i}:
Clinical variables:
{_format_lines(row, CLINICAL_COLS)}{micro_block}{macro_block}

True iAUC120: {row.iloc[TRUE_IAUC_COL]}
""".strip()

def make_examples(
    train_df: pd.DataFrame,
    input_mode: str,
    n_examples: int = 3,
    use_microbiome: bool = True,
) -> str:
    return "\n\n".join(
        row_to_example(row, input_mode, i + 1, use_microbiome=use_microbiome)
        for i, (_, row) in enumerate(train_df.head(n_examples).iterrows())
    )
def make_target_summary(train_df: pd.DataFrame) -> str:
    y = train_df.iloc[:, TRUE_IAUC_COL].astype(float)

    return (
        f"Training-set iAUC120 scale:\n"
        f"- minimum: {y.min():.1f}\n"
        f"- 25th percentile: {y.quantile(0.25):.1f}\n"
        f"- median: {y.median():.1f}\n"
        f"- 75th percentile: {y.quantile(0.75):.1f}\n"
        f"- maximum: {y.max():.1f}\n"
        f"Values around 100 are very low in this dataset; high responders can have values of several thousand."
    )

def make_quantile_examples(
    train_df: pd.DataFrame,
    input_mode: str,
    quantiles=(0.05, 0.25, 0.50, 0.75, 0.95),
) -> str:
    y = train_df.iloc[:, TRUE_IAUC_COL].astype(float)

    selected_indices = []
    for q in quantiles:
        target_value = y.quantile(q)
        idx = (y - target_value).abs().idxmin()
        selected_indices.append(idx)

    # Remove duplicates while preserving order
    selected_indices = list(dict.fromkeys(selected_indices))

    example_rows = train_df.loc[selected_indices]

    return "\n\n".join(
        row_to_example(row, input_mode, i + 1)
        for i, (_, row) in enumerate(example_rows.iterrows())
    )

def row_to_prompt(
    row: pd.Series,
    input_mode: str,
    examples_text: str = "",
    target_summary_text: str = "",
    use_microbiome: bool = True,
) -> str:
    use_image = input_mode in IMAGE_MODES
    use_macros = input_mode in MACRO_MODES

    modality_sentence = (
        "You are given a food image and structured participant information."
        if use_image
        else "You are given structured participant and meal information."
    )

    macro_block = ""
    if use_macros:
        macro_block = f"""

True meal macronutrients:
{_format_lines(row, TRUE_MACRO_COLS)}
""".rstrip()

    examples_block = ""
    if examples_text:
        examples_block = f"""
Use the following training examples only as calibration references for the target scale:

{examples_text}
""".strip()

    image_instruction = (
        "Before predicting, internally estimate the visible meal composition from the image, "
        "including approximate calories, carbohydrates, fat, protein, and portion size. "
        "Then combine this inferred meal composition with the clinical and microbiome variables."
        if use_image
        else
        "Use the provided clinical, microbiome, and meal variables to predict the glucose response."
    )

    prompt = f"""
{modality_sentence}
Your task is to predict the participant-specific 2-hour postprandial glucose incremental area under the curve, iAUC120.

iAUC120 is a raw area value from this dataset, not a probability, not a glucose concentration, and not a normalized score.

{target_summary_text}

{examples_block}

Now predict the current sample.

{image_instruction}

Clinical variables:
{_format_lines(row, CLINICAL_COLS)}

Microbiome features:
{_format_lines(row, MICROBIOME_COLS)}{macro_block}

Important:
- iAUC120 is a raw area-under-the-curve value from this dataset.
- Values below 500 are very low and should only be used for almost flat glucose responses.
- Low responses are often around 500-2000.
- Moderate responses are often around 2000-5000.
- High responses are often around 5000-8000.
- Very high responses can exceed 8000.
- A meal with moderate carbohydrates can still produce a high iAUC120 depending on the participant.
- The prediction must be specific to the current meal and participant.
- Do not return a default value such as 10, 100, 123, 350, 1000, or 1234.
- Include only a brief rationale, not step-by-step reasoning.
- Return only valid JSON with one key for the prediction and one key for the rationale.

Return exactly this JSON structure:
{{
  "predicted_iAUC120": <raw numeric iAUC120 value>,
  "rationale": "<one short sentence mentioning meal composition and participant-specific factors>"
}}
""".strip()

    return prompt


# def row_to_prompt(row: pd.Series, input_mode: str, examples_text: str = "") -> str:
#     use_image = input_mode in IMAGE_MODES
#     use_macros = input_mode in MACRO_MODES
#
#     modality_sentence = (
#         "You are given a food image and structured participant information."
#         if use_image
#         else "You are given structured participant and meal information."
#     )
#
#     macro_block = ""
#     if use_macros:
#         macro_block = f"""
#
# True meal macronutrients:
# {_format_lines(row, TRUE_MACRO_COLS)}
# """.rstrip()
#
#     examples_block = ""
#     if examples_text:
#         examples_block = f"""
# Use these training examples as calibration references:
#
# {examples_text}
#
# Now predict the current sample.
# """.strip()
#
#     prompt = f"""
# {modality_sentence}
# Your task is to predict the 2-hour postprandial glucose incremental area under the curve, iAUC120.
#
# {examples_block}
#
# Before predicting, internally estimate the visible meal composition from the image, including approximate calories, carbohydrates, fat, protein, and portion size. Then combine this with the clinical and microbiome variables.
#
# Clinical variables:
# {_format_lines(row, CLINICAL_COLS)}
#
# Microbiome features:
# {_format_lines(row, MICROBIOME_COLS)}{macro_block}
#
# Important:
# - iAUC120 is a raw glucose-response area value, not a probability, not a correlation, and not a normalized score.
# - Explain your reasoning.
# - Return exactly one JSON object with this schema:
# {{"predicted_iAUC120": <number>,
# "rationale": "<one short sentence explaining the main factors used>"}}
# """.strip()
#
#     return prompt



def extract_prediction(raw_response) -> float:
    """Extract predicted_iAUC120 robustly, avoiding the 120 in the key name."""
    if raw_response is None:
        return math.nan

    text = str(raw_response).strip()

    # Remove markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # First try JSON parsing.
    try:
        obj = json.loads(text)
        if "predicted_iAUC120" in obj:
            return float(obj["predicted_iAUC120"])
    except Exception:
        pass

    # Then try exact key-value extraction.
    match = re.search(
        r'"?predicted_iAUC120"?\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
        text,
    )
    if match:
        return float(match.group(1))

    # Fallback: take the last number, not the first, because the key contains 120.
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if numbers:
        return float(numbers[-1])

    return math.nan


# ---------------------------------------------------------------------
# Qwen2.5-VL backend
# ---------------------------------------------------------------------

# def load_qwen(model_id: str):
#     from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
#
#     processor = AutoProcessor.from_pretrained(
#         model_id,
#         min_pixels=128 * 28 * 28,
#         max_pixels=256 * 28 * 28,
#     )
#
#     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         attn_implementation="eager",
#         low_cpu_mem_usage=True,
#     ).eval()
#
#     print("Qwen device map:", getattr(model, "hf_device_map", None))
#     return model, processor

def load_qwen(model_id: str):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,   # important for RTX 6000 / 2080 Ti
        device_map="auto",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    print("Qwen3.6 device map:", getattr(model, "hf_device_map", None))
    return model, processor


# def predict_qwen(model, processor, prompt: str, image_path: Path | None = None) -> str:
#     from qwen_vl_utils import process_vision_info
#
#     if image_path is None:
#         messages = [
#             {
#                 "role": "user",
#                 "content": [{"type": "text", "text": prompt}],
#             }
#         ]
#         text = processor.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#         inputs = processor(
#             text=[text],
#             padding=True,
#             return_tensors="pt",
#         ).to("cuda")
#     else:
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": str(image_path)},
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ]
#         text = processor.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         ).to("cuda")
#
#     inputs = {k: v.to("cuda") for k, v in inputs.items()}
#
#     with torch.no_grad():
#         generated_ids = model.generate(
#             **inputs,
#             max_new_tokens=128,
#             do_sample=False,
#         )
#
#     generated_ids_trimmed = [
#         out_ids[len(in_ids):]
#         for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False,
#     )[0]
#     return output_text

def predict_qwen(model, processor, prompt: str, image_path: Path | None = None) -> str:
    if image_path is None:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
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
        if min_num <= i * j <= max_num
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
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_internvl_image(image_file: Path, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def load_internvl(model_id: str, load_8bit: bool = False):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
    )

    if load_8bit:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
            device_map="auto",
        ).eval()
    else:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
            device_map="auto",
        ).eval()

    print("InternVL device map:", getattr(model, "hf_device_map", None))
    print("InternVL dtype:", next(model.parameters()).dtype)
    return model, tokenizer


def predict_internvl(model, tokenizer, prompt: str, image_path: Path | None = None) -> str:
    generation_config = dict(
        max_new_tokens=64,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )

    if image_path is None:
        question = prompt
        pixel_values = None
    else:
        pixel_values = load_internvl_image(image_path, max_num=12)
        pixel_values = pixel_values.to(device="cuda", dtype=torch.bfloat16)
        question = "<image>\n" + prompt

    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response


# ---------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------

def run(args):
    df = pd.read_csv(args.txt, sep=r"\s+", header=None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    examples_text = ""
    target_summary_text = ""
    if args.train_txt is not None and args.n_examples > 0:
        train_df = load_txt(args.train_txt)
        examples_text = make_examples(train_df, args.input_mode, n_examples=args.n_examples)
        print(f"Using {args.n_examples} few-shot examples from: {args.train_txt}")
        # examples_text = make_quantile_examples(train_df, args.input_mode)
        target_summary_text = make_target_summary(train_df)

    if args.backend == "qwen":
        model, processor_or_tokenizer = load_qwen(args.model_id)
        predict_fn = lambda prompt, img=None: predict_qwen(
            model,
            processor_or_tokenizer,
            prompt,
            image_path=img,
        )
    elif args.backend == "internvl":
        model, processor_or_tokenizer = load_internvl(args.model_id, load_8bit=args.load_8bit)
        predict_fn = lambda prompt, img=None: predict_internvl(
            model,
            processor_or_tokenizer,
            prompt,
            image_path=img,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    rows = []
    use_image = args.input_mode in IMAGE_MODES

    for idx, row in df.iterrows():
        image_path = resolve_image_path(row.iloc[0], args.image_root) if use_image else None
        # prompt = row_to_prompt(row, args.input_mode, examples_text=examples_text)
        prompt = row_to_prompt(
            row,
            args.input_mode,
            examples_text=examples_text,
            target_summary_text=target_summary_text,
        )

        if use_image and not image_path.exists():
            raw_response = f"ERROR: image not found: {image_path}"
            pred = np.nan
        else:
            try:
                raw_response = predict_fn(prompt, image_path)
                pred = extract_prediction(raw_response)
            except Exception as e:
                raw_response = f"ERROR: {repr(e)}"
                pred = np.nan

        true_iauc = float(row.iloc[TRUE_IAUC_COL])
        image_path_str = str(row.iloc[0]) if use_image else ""

        # Matches your usual evaluation format, with optional debugging columns.
        subject_match = re.search(r"CGMacros-(\d+)", str(row.iloc[0]))
        subject_id = subject_match.group(1) if subject_match else ""
        out_row = {
            "subject_id": subject_id,
            "y_true": true_iauc,
            "y_pred": pred,
        }
        if args.keep_debug_cols:
            out_row.update({
                "row_idx": idx,
                "image_path": image_path_str,
                "image_id": str(row.iloc[1]),
                "input_mode": args.input_mode,
                "raw_response": raw_response,
            })
        rows.append(out_row)

        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"[{idx + 1}/{len(df)}] true={true_iauc:.3f}, pred={pred}, response={raw_response}")

        if args.max_rows is not None and len(rows) >= args.max_rows:
            break

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
    parser.add_argument(
        "--input_mode",
        choices=["image_clinical", "image_clinical_macros", "clinical_macros", "clinical_only"],
        default="image_clinical",
        help="Controls whether the VLM receives an image and/or true macronutrients.",
    )
    parser.add_argument("--load_8bit", action="store_true", help="Only used for InternVL backend")
    parser.add_argument("--keep_debug_cols", action="store_true", help="Save raw VLM responses and row metadata")
    parser.add_argument("--max_rows", type=int, default=None, help="Debug option: stop after N rows")
    parser.add_argument("--train_txt", default=None, help="Optional train.txt used for few-shot calibration examples")
    parser.add_argument("--n_examples", type=int, default=0,
                        help="Number of first rows from train.txt to use as examples")
    parser.add_argument(
        "--no_microbiome",
        dest="use_microbiome",
        action="store_false",
        help="Exclude microbiome variables from prompts and few-shot examples.",
    )
    parser.set_defaults(use_microbiome=True)
    args = parser.parse_args()
    run(args)
