import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniTransformerLM(nn.Module):
    """
    Un pequeño modelo Transformer autoregresivo para modelado de lenguaje.
    """

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, dropout=0.1, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.size()
        device = input_ids.device
        positions = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.transformer(x, src_key_padding_mask=(~attention_mask.bool()) if attention_mask is not None else None)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
