import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

model_id = "OpenGVLab/InternVL3-2B-Instruct"
image_path = "../cgmacros/CGMacros/CGMacros-022/photos/00000010-PHOTO-2021-3-29-13-37-0.jpg"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=False,
)

model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_flash_attn=False,
    device_map={"": 0},
).eval()

transform = build_transform(input_size=448)
image = Image.open(image_path).convert("RGB")
pixel_values = transform(image).unsqueeze(0)

pixel_values = pixel_values.to(device="cuda", dtype=torch.bfloat16)

print("model dtype:", next(model.parameters()).dtype)
print("pixel dtype:", pixel_values.dtype)
print("pixel shape:", pixel_values.shape)

generation_config = dict(
    max_new_tokens=64,
    do_sample=False,
    num_beams=1,
    pad_token_id=tokenizer.eos_token_id,
)

question = "<image>\nDescribe this image in one short sentence."

with torch.no_grad():
    response = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
    )

print("RESPONSE:", repr(response))