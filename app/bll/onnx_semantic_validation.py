from __future__ import annotations

from typing import List, Tuple
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class OnnxSemanticValidator:
    """
    Validação de busca semântica usando ONNX Runtime.
    Projetado para ser chamado após a conversão do modelo.
    """

    def __init__(
        self,
        onnx_path: str,
        model_path: str,
        max_length: int = 512,
    ):
        self.onnx_path = onnx_path
        self.model_path = model_path
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        self.session = self._create_session()

    # =========================
    # Inicialização ORT
    # =========================

    def _create_session(self) -> ort.InferenceSession:
        providers = ort.get_available_providers()

        if "CUDAExecutionProvider" in providers:
            selected = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            selected = ["CPUExecutionProvider"]

        return ort.InferenceSession(
            self.onnx_path,
            providers=selected,
        )

    # =========================
    # Embedding
    # =========================

    def embed(self, texts: List[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        outputs = self.session.run(
            ["sentence_embedding"],
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            },
        )

        embeddings = outputs[0]
        return self._l2_normalize(embeddings)

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norm, eps, None)

    # =========================
    # Busca semântica
    # =========================

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b.T)[0, 0])

    def semantic_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        query_emb = self.embed([query])
        doc_embs = self.embed(documents)

        scores = [
            self.cosine_similarity(query_emb, doc_embs[i : i + 1])
            for i in range(len(documents))
        ]

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_k]
