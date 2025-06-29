import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data.collator import CustomDataCollator
from models.transformer_lm import MiniTransformerLM
from datasets import load_dataset
from torch.nn.functional import cross_entropy
import math

# Configuración
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_model = "gpt2"
batch_size = 4
num_epochs = 5
lr = 5e-4

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(vocab_model)
tokenizer.pad_token = tokenizer.eos_token

# Dataset
dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
collator = CustomDataCollator(tokenizer, mlm_probability=0.15)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

# Modelo
model = MiniTransformerLM(
    vocab_size=tokenizer.vocab_size,
    d_model=256,
    n_heads=4,
    n_layers=4,
    max_len=128
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Métricas de diversidad
def distinct_n(seqs, n):
    total_ngrams = 0
    unique_ngrams = set()
    for seq in seqs:
        tokens = seq.tolist()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        ngrams = list(ngrams)
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0

# Entrenamiento
model.train()
for epoch in range(1, num_epochs + 1):
    total_loss = 0
    all_labels = []

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_labels.extend([l[l != -100] for l in labels])  # solo tokens válidos

        if (step + 1) % 100 == 0:
            print(f"[Epoch {epoch} | Step {step+1}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    ppl = math.exp(avg_loss)
    d1 = distinct_n(all_labels, 1)
    d2 = distinct_n(all_labels, 2)

    print("=" * 30)
    print(f"\nEpoch {epoch}")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")
    print(f"Distinct-1: {d1:.4f}")
    print(f"Distinct-2: {d2:.4f}")
    print("=" * 30)
