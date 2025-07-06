"""
Este archivo define el collator personalizado que aplica enmascaramiento
dinámico o estático a los datos antes de entrenar el modelo.
"""

"""
Explica brevemente el trade-off memoria vs. variedad de contextos

Se refiere a la relación entre el tamaño de los chunks de texto y la cantidad de memoria utilizada durante el entrenamiento del modelo.
Como se mencinó en el script 'clean_corpus' :
- Chunks más pequeños permiten mayor diversidad y menos uso de memoria, pero pueden perder coherencia contextual
- Chunks más grandes pueden capturar mejor el contexto, pero requieren más memoria
"""

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch
import random

class CustomDataCollator:
    """
    Collator que agrega máscaras a los tokens:
    - Masking dinámico: oculta tokens individuales aleatoriamente
    - Masking estático: oculta un porcentaje de los chunks completos
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_probability: float = 0.15, static_masking: bool = False):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.static_masking = static_masking
        self.mask_token_id = tokenizer.mask_token_id or tokenizer.unk_token_id
     
    def mask_dynamic(self, input_ids, labels):
        """
        Enmascaramiento dinámico por token (15%)
        """
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
            for val in labels
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # No calcula pérdida donde no hay máscara
        input_ids[masked_indices] = self.mask_token_id
        return input_ids, labels

    def mask_static(self, input_ids, labels):
        """
        Enmascaramiento estático: oculta un 10% de los chunks completos
        """
        batch_size = input_ids.size(0)
        num_mask = max(1, int(0.1 * batch_size))
        mask_indices = random.sample(range(batch_size), num_mask)
        for idx in mask_indices:
            input_ids[idx] = self.mask_token_id
            
        # Para los no enmascarados, no se calcula pérdida
        for idx in range(batch_size):
            if idx not in mask_indices:
                labels[idx] = -100
        return input_ids, labels
    
    def __call__(self, examples):
        # Extraer los textos del batch
        texts = [ex["text"] for ex in examples]
        encoding = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        input_ids = encoding["input_ids"]
        labels = input_ids.clone()

        if self.static_masking:
            input_ids, labels = self.mask_static(input_ids, labels)
        else:
            input_ids, labels = self.mask_dynamic(input_ids, labels)

        # Devuelve los tensores listos para el modelo
        return {"input_ids": input_ids, "labels": labels, "attention_mask": encoding["attention_mask"]}