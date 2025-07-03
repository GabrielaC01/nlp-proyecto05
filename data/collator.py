"""
Este archivo define el collator personalizado que aplica enmascaramiento
dinámico o estático a los datos antes de entrenar el modelo.
"""

from transformers import PreTrainedTokenizerBase
import torch
import random

class CustomDataCollator:
    """
    Collator que agrega máscaras a los tokens:
    - Si static_masking es True, puede ocultar toda la línea.
    - Si static_masking es False, oculta tokens individuales aleatoriamente.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_probability: float = 0.15, static_masking: bool = False):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.static_masking = static_masking
        self.mask_token_id = tokenizer.mask_token_id or tokenizer.unk_token_id

    def __call__(self, examples):
        # Extraer los textos del batch
        texts = [ex["text"] for ex in examples]
        encoding = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        input_ids = encoding["input_ids"]
        labels = input_ids.clone()

        if self.static_masking:
            # Enmascaramiento estático: oculta toda la línea con cierta probabilidad
            for i in range(len(input_ids)):
                if random.random() < self.mlm_probability:
                    input_ids[i] = self.mask_token_id
        else:
            # Enmascaramiento dinámico por token, como BERT
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
                for val in labels
            ]
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # No calcula pérdida donde no hay máscara
            input_ids[masked_indices] = self.mask_token_id

        # Devuelve los tensores listos para el modelo
        return {"input_ids": input_ids, "labels": labels, "attention_mask": encoding["attention_mask"]}
