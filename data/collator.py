from transformers import PreTrainedTokenizerBase
import torch
import random

class CustomDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_probability: float = 0.15, static_masking: bool = False):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.static_masking = static_masking
        self.mask_token_id = tokenizer.mask_token_id or tokenizer.unk_token_id

    def __call__(self, examples):
        # Extraer textos
        texts = [ex["text"] for ex in examples]
        encoding = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        input_ids = encoding["input_ids"]
        labels = input_ids.clone()

        if self.static_masking:
            # Enmascarar líneas completas (por ejemplo, 1 de cada 6)
            for i in range(len(input_ids)):
                if random.random() < self.mlm_probability:
                    input_ids[i] = self.mask_token_id
        else:
            # Enmascaramiento dinámico tipo BERT
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
                for val in labels
            ]
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # Ignora pérdida en tokens no enmascarados
            input_ids[masked_indices] = self.mask_token_id

        return {"input_ids": input_ids, "labels": labels, "attention_mask": encoding["attention_mask"]}
