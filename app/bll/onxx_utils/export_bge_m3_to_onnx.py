#export_bge_m3_to_onnx.py
from __future__ import annotations

from math import exp
import os
from re import S
from threading import local
from typing import Dict, Tuple
import shutil
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import onnx
from bll.onxx_utils.onnx_semantic_validation import OnnxSemanticValidator
from common import print_with_time
import localconfig
import bll.onxx_utils.dense_embedding_wrapper as dense_embedding_wrapperModule

### EXPORTA UM MODELO BGE-M3 DENSO PARA ONNX ###
### O ONXX É MUITO MAIS RAPIDO PARA INFERÊNCIA EM PRODUÇÃO E CPU###


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

    def __init__(self): 
        import localconfig      
        self.model_path_source: str = str(localconfig.getModelPath())
        self.max_length: int = int(localconfig.get("max_length"))        
        if self.max_length > 1024:
            raise RuntimeError(f"[AVISO] max_length não pode ser maior que 1024.")
            
        self.dest_path = f"{self.model_path_source}-dense-onnx/"        
        self.model_filename_destination: str = f"{self.dest_path}/bge-m3-dense.onnx"        
        self.opset_version: int = 18        

    def load_model(self) -> Tuple[AutoTokenizer, nn.Module]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path_source,
            local_files_only=True,
        )

        base_model = AutoModel.from_pretrained(
            self.model_path_source,
            local_files_only=True,
            torch_dtype=torch.float32,            
        )

        base_model.eval()

        wrapped_model =  dense_embedding_wrapperModule.DenseEmbeddingTorchWrapper(base_model)
        wrapped_model.eval()

        return tokenizer, wrapped_model


    def build_dummy_inputs(
            self,
            tokenizer: AutoTokenizer            
        ) -> Dict[str, torch.Tensor]:
            # Cria batch fixo com textos dummy iguais (padding até max_length)
            dummy_texts = ["Dummy input"] 
            encoded = tokenizer(
                dummy_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ) # type: ignore
            return {
                "input_ids": encoded["input_ids"],          
                "attention_mask": encoded["attention_mask"],
            }
    
    def export(self) -> str:
            tokenizer, model = self.load_model()
            dummy_inputs = self.build_dummy_inputs(tokenizer)

            print(f"Exportando BGE-M3 ONNX com batch FIXO=1 e seq DINÂMICA (truncada a <= {self.max_length})")                   

            os.makedirs(os.path.dirname(self.model_filename_destination) or ".", exist_ok=True)

            with torch.no_grad():
                torch.onnx.export(
                    model,
                    args=(
                        dummy_inputs["input_ids"],
                        dummy_inputs["attention_mask"],
                    ),
                    f=self.model_filename_destination,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["sentence_embedding"],
                    dynamic_axes={
                        "input_ids": {1: "seq_len"},
                        "attention_mask": {1: "seq_len"},
                    }, 
                    opset_version=self.opset_version,
                    do_constant_folding=True,
                )

            return self.model_filename_destination
    # =========================
    # Limpa o diretório destino
    def clean_destination_directory(self, dst_dir:str) -> None:    
        if not os.path.exists(dst_dir):
            return        
        # ===== LIMPA O DIRETÓRIO DESTINO =====
        for fname in os.listdir(dst_dir):
            fpath = os.path.join(dst_dir, fname)
            try:
                if os.path.isfile(fpath) or os.path.islink(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
            except Exception as e:
                print(f"[AVISO] Falha ao remover {fpath}: {e}")

    #exporta arquivos secundários necessários (tokenizer/configs)
    def export_secondary_files(self) -> None:
        """
        Copia arquivos necessários do modelo HF (tokenizer/configs)
        para a pasta do modelo ONNX, facilitando deploy.
        Antes da cópia, limpa completamente o diretório destino.
        """
        src_dir = self.model_path_source
        dst_dir = os.path.dirname(self.model_filename_destination)

        # Garante que o diretório existe
        os.makedirs(dst_dir, exist_ok=True)

        # ===== COPIA OS ARQUIVOS NECESSÁRIOS =====
        files_to_copy = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "config.json",
            "modules.json",
            "sentence_bert_config.json",
            "config_sentence_transformers.json",
        ]

        for fname in files_to_copy:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)

            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"[AVISO] Arquivo não encontrado e ignorado: {src}")
    # =========================

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

def _hf_embed_cls_l2(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    """
    Gera embeddings com o BGE-M3 original (HF) usando:
    - CLS pooling: last_hidden_state[:, 0, :]
    - L2 normalize
    Retorna: (batch, dim) float32
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ) # type: ignore
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :]  # (batch, dim)

        # L2 normalize em torch
        cls = cls / torch.linalg.norm(cls, dim=-1, keepdim=True).clamp(min=1e-12)

    return cls.detach().cpu().numpy().astype("float32")


def _cosine_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Assume vetores já L2-normalizados.
    Retorna cosine = dot(query, docs).
    """
    return doc_vecs @ query_vec  # (n_docs,)


def validate_hf_semantic_search(cfg) -> List[Tuple[str, float]]:
    """
    Executa o mesmo teste do ONNX, mas usando o BGE-M3 original (Transformers).
    Mantém o mesmo comportamento do seu pipeline antigo (CLS + L2).
    """
    # Você pode colocar o tipo de cfg como BgeM3OnnxExporter se quiser
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path_source, local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        cfg.model_path_source,
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    documents = [
        "O gato está dormindo no sofá",
        "Um cachorro corre no parque",
        "Programação em Python para ciência de dados",
        "Machine learning e inteligência artificial",
        "Receita de bolo de chocolate",
    ]
    query = "aprendizado de máquina"

    # Embeddings
    q = _hf_embed_cls_l2(model, tokenizer, [query], cfg.max_length, device)[0]      # (dim,)
    d = _hf_embed_cls_l2(model, tokenizer, documents, cfg.max_length, device)      # (n_docs, dim)

    scores = _cosine_scores(q, d)  # (n_docs,)

    results = sorted(zip(documents, scores.tolist()), key=lambda x: x[1], reverse=True)

    print(f"\n🔎 Resultados da busca semântica (HF/original):\n")
    for text, score in results:
        print(f"{score:.4f} | {text}")

    return results

def validate_onnx_semantic_search(cfg: BgeM3OnnxExporter):
    validator = OnnxSemanticValidator(
        onnx_path=cfg.model_filename_destination,
        model_path=cfg.model_path_source,
        max_length=cfg.max_length  # use o mesmo da exportação
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

    print(f"\n🔎 Resultados da busca semântica (ONNX):\n")
    for text, score in results:
        print(f"{score:.4f} | {text}")

    input("\nPressione Enter para continuar...")   

# =========================
# Execução direta
# =========================

def execute():
    print_with_time("Convertendo modelo  BGE-M3 para ONNX...")    

    exporter = BgeM3OnnxExporter()    
    exporter.clean_destination_directory(exporter.dest_path)    
    exporter.model_filename_destination = f"{exporter.dest_path}/bge-m3-dense.onnx"
    
    onnx_path = exporter.export()
    onnx.checker.check_model(onnx_path)
    print_with_time(f"Modelo ONNX gerado com sucesso em: {onnx_path}")

    exporter.export_secondary_files()
    validate_onnx_semantic_search(exporter)   
    validate_hf_semantic_search(exporter)
          

    print_with_time("ONNX válidado com sucesso")    