from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import onnx
from bll.onnx_semantic_validation import OnnxSemanticValidator

### EXPORTA UM MODELO BGE-M3 DENSO PARA ONNX ###
### O ONXX É MUITO MAIS RAPIDO PARA INFERÊNCIA EM PRODUÇÃO E CPU###

# =========================
# Configuração
# =========================

@dataclass(frozen=True)
class ExportConfig:
    model_id: str = "/opt/modelos/bge-m3"
    max_length: int = 1024
    opset_version: int = 18
    output_path: str = "/opt/modelos/bge-m3-dense-onnx/bge-m3-dense.onnx"


# =========================
# Wrapper do modelo
# =========================

class DenseEmbeddingWrapper(nn.Module):
    """
    Wrapper para exportação ONNX:
    - Encoder do bge-m3
    - Mean pooling com attention_mask
    - L2 normalization

    Saída:
      embedding: (batch_size, hidden_dim)
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()

    @staticmethod
    def mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @staticmethod
    def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)
        return x / norm

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        normalized = self.l2_normalize(pooled)
        return normalized


# =========================
# Exportador ONNX
# =========================

class BgeM3OnnxExporter:
    """
    Responsável exclusivamente por:
    - Carregar o bge-m3
    - Envolver com pooling + normalização
    - Exportar para ONNX compatível com CPU/GPU
    """

    def __init__(self, config: ExportConfig):
        self.config = config

    def load_model(self) -> Tuple[AutoTokenizer, nn.Module]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            local_files_only=True,
        )

        base_model = AutoModel.from_pretrained(
            self.config.model_id,
            local_files_only=True,
            torch_dtype=torch.float32,            
        )

        base_model.eval()

        wrapped_model = DenseEmbeddingWrapper(base_model)
        wrapped_model.eval()

        return tokenizer, wrapped_model


    def build_dummy_inputs(
        self,
        tokenizer: AutoTokenizer,
    ) -> Dict[str, torch.Tensor]:
        dummy_text = "Dummy input for ONNX export"
        encoded = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
        ) # type: ignore
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def export(self) -> str:
        tokenizer, model = self.load_model()
        dummy_inputs = self.build_dummy_inputs(tokenizer)

        os.makedirs(os.path.dirname(self.config.output_path) or ".", exist_ok=True)

        with torch.no_grad():
            torch.onnx.export(
                model,
                args=(
                    dummy_inputs["input_ids"],
                    dummy_inputs["attention_mask"],
                ),
                f=self.config.output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["sentence_embedding"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "sentence_embedding": {0: "batch_size"},
                },
                opset_version=self.config.opset_version,
                do_constant_folding=True,
            )

        return self.config.output_path


def validate_onnx_semantic_search():
    validator = OnnxSemanticValidator(
        onnx_path="/opt/modelos/bge-m3-dense-onnx/bge-m3-dense.onnx",
        model_path="/opt/modelos/bge-m3",
        max_length=1024,  # use o mesmo da exportação
    )

    documents = [
        "O gato está dormindo no sofá",
        "Um cachorro corre no parque",
        "Programação em Python para ciência de dados",
        "Machine learning e inteligência artificial",
        "Receita de bolo de chocolate",
    ]

    query = "aprendizado de máquina"

    results = validator.semantic_search(query, documents)

    print("\n🔎 Validação de busca semântica (ONNX):\n")
    for text, score in results:
        print(f"{score:.4f} | {text}")


# =========================
# Execução direta
# =========================

def execute():
    cfg = ExportConfig()

    exporter = BgeM3OnnxExporter(cfg)
    onnx_path = exporter.export()    

    print(f"Modelo ONNX gerado com sucesso em: {onnx_path}")

    onnx.checker.check_model(onnx_path)
    
    validate_onnx_semantic_search()

    print("ONNX válidado com sucesso")    
