#dense_embedding_wrapper.py
import torch
import torch.nn as nn

# =========================
# Wrapper do modelo
# =========================

class DenseEmbeddingTorchWrapper(nn.Module):
    """
    Compatível com o ONNX dinâmico atual:
    - encoder HF
    - CLS pooling (token 0)
    - L2 normalization
    Retorna: (batch, hidden_dim)
    """
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder.eval()

    @staticmethod
    def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)
        return x / norm

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.l2_normalize(cls)

