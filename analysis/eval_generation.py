"""
Este script carga el modelo MiniTransformerLM ya entrenado y genera texto corto.
Calcula las métricas de diversidad distinct-1 y distinct-2 sobre los textos generados.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.transformer_lm import MiniTransformerLM

# Configuración
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_model = "gpt2"
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "checkpoints", "minitransformer.pt")
num_samples = 5
top_k = 10   
temperature = 1.0

# Tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(vocab_model)
tokenizer.pad_token = tokenizer.eos_token

model = MiniTransformerLM(
    vocab_size=tokenizer.vocab_size,
    d_model=256,
    n_heads=4,
    n_layers=4,
    max_len=128
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Genera texto a partir de un prompt inicial
def generar_texto(prompt="", max_len=100, top_k=10, temperature=1.0):
    if not prompt.strip():
        prompt = tokenizer.eos_token

    ids = tokenizer.encode(prompt, return_tensors="pt").to(device).long()
    attn_mask = torch.ones_like(ids)

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(ids, attention_mask=attn_mask)

        next_logits = logits[:, -1, :] / temperature
        top_k_probs, top_k_indices = torch.topk(next_logits, k=top_k, dim=-1)
        probs = torch.softmax(top_k_probs, dim=-1)
        next_token = top_k_indices[0, torch.multinomial(probs, num_samples=1)]
        next_token = next_token.view(1, 1)

        ids = torch.cat([ids, next_token], dim=1)
        attn_mask = torch.ones_like(ids)

    return tokenizer.decode(ids[0], skip_special_tokens=True)

# Calcula métricas distinct-n
def distinct_n(seqs, n):
    total_ngrams = 0
    unique_ngrams = set()
    for text in seqs:
        tokens = text.split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngrams = list(ngrams)
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0

# Ejecuta la generación y calcula diversidad
if __name__ == "__main__":
    resultados = []

    resultados_sin_prompt = []
    resultados_con_prompt = []

    print("\nGeneración SIN prompt:")
    for _ in range(num_samples):
        texto = generar_texto("", max_len=128, top_k=top_k, temperature=temperature)
        print("-" * 30)
        print(texto)
        resultados_sin_prompt.append(texto)

    print("\nGeneración CON prompt 'The little cat':")
    for _ in range(num_samples):
        texto = generar_texto("The little cat", max_len=50, top_k=top_k, temperature=temperature)
        print("-" * 30)
        print(texto)
        resultados_con_prompt.append(texto)

    d1_sin = distinct_n(resultados_sin_prompt, 1)
    d2_sin = distinct_n(resultados_sin_prompt, 2)

    d1_con = distinct_n(resultados_con_prompt, 1)
    d2_con = distinct_n(resultados_con_prompt, 2)

    print("\nMétricas de diversidad SIN prompt:")
    print(f"Distinct-1: {d1_sin:.4f}")
    print(f"Distinct-2: {d2_sin:.4f}")

    print("\nMétricas de diversidad CON prompt:")
    print(f"Distinct-1: {d1_con:.4f}")
    print(f"Distinct-2: {d2_con:.4f}")

