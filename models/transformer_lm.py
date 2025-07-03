"""
Este archivo define un modelo Transformer pequeño (MiniTransformerLM)
para modelado de lenguaje autoregresivo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniTransformerLM(nn.Module):
    """
    Pequeño modelo Transformer autoregresivo para predecir el siguiente token en una secuencia.
    Tiene embedding de tokens, embedding posicional, varias capas Transformer
    y una capa final que proyecta a las clases del vocabulario.
    """

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, dropout=0.1, max_len=512):
        super().__init__()

        # Embedding de tokens y posiciones
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Capas Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Normalización y capa final
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # Calcula posiciones y embeddings
        B, T = input_ids.size()
        device = input_ids.device
        positions = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)

        # Suma embeddings de tokens + posiciones
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # Pasa por las capas Transformer
        x = self.transformer(x, src_key_padding_mask=(~attention_mask.bool()) if attention_mask is not None else None)

        # Normalización y capa de salida
        x = self.ln_f(x)
        logits = self.head(x)

        return logits
