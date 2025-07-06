"""
Este archivo define un modelo Transformer pequeño (MiniTransformerLM)
para modelado de lenguaje autoregresivo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniTransformerLM(nn.Module):

    """
    Añade docstrings que expliquen cada componente (embeddings, capas de atención, feed-forward)
 
    Pequeño modelo Transformer autoregresivo para predecir el siguiente token en una secuencia.

    Componentes:
    1) Embeddings:
        - self.token_emb: Convierte los índices de tokens en vectores densos de dimensión d_model
        - self.pos_emb: Codifica la posición de cada token en la secuencia, permitiendo al modelo distinguir el orden

    2) Capas de atención (self.transformer):
        - Conjunto de 4 capas Transformer Encoder, cada una con mecanismos de atención multi-heads
        - Permiten que el modelo relacione cada token con otros tokens de la secuencia, capturando dependencias a largo plazo

    3) Feed-forward:
        - input_ids: Tensor (batch, seq_len) con los índices de los tokens
        - attention_mask: Tensor opcional que indica qué posiciones deben ser atendidas (1) o ignoradas (0)
        - Dentro de cada capa Transformer hay una red feed-forward que transforma las representaciones de los tokens de manera no lineal

    4) Normalización y salida:
        - self.ln_f: Normaliza la salida final del Transformer
        - self.head: Proyecta la representación final de cada token al espacio del vocabulario para predecir el siguiente token
    """

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, dropout=0.1, max_len=512):
        super().__init__()

        # Embedding de tokens 
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Embedding posicional
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

        # Normalización final 
        self.ln_f = nn.LayerNorm(d_model)
        # Capa de salida
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
