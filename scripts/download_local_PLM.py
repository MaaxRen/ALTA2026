from transformers import AutoTokenizer, AutoModel
import os

name = "Qwen/Qwen3-Embedding-8B"
out_dir = "../ALTA2026/resources/pretrained_model/qwen3-embedding-8b"
os.makedirs(out_dir, exist_ok=True)

tok = AutoTokenizer.from_pretrained(name)
tok.save_pretrained(out_dir)

model = AutoModel.from_pretrained(name)
model.save_pretrained(out_dir)

print("Saved to:", out_dir)