import torch
from transformers import AutoTokenizer, AutoModel

model_id = "OpenGVLab/InternVL3-2B-Instruct"

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

print("tokenizer:", tokenizer.name_or_path)
print("model dtype:", next(model.parameters()).dtype)
print("model device:", next(model.parameters()).device)

generation_config = dict(
    max_new_tokens=64,
    do_sample=False,
)

response, history = model.chat(
    tokenizer,
    None,
    "Hello, who are you?",
    generation_config,
    history=None,
    return_history=True,
)

print("RESPONSE:", repr(response))