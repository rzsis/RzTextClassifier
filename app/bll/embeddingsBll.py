# embeddingsBll.py
import numpy as np
from db_utils import Session
from typing import  Dict, List, Optional,Union
from common import print_with_time
import logger
from transformers import AutoTokenizer,AutoModel
import torch
import torch.nn as nn
import bll.onxx_utils.dense_embedding_wrapper as dense_embedding_wrapperModule

bllEmbeddings = None



def initBllEmbeddings(session=Session):
    global bllEmbeddings
    if bllEmbeddings is None:
        try:
            from main import localconfig
            bllEmbeddings = EmbeddingsBll()
            bllEmbeddings.load_model_and_tokenizer()
        except Exception as e:
            raise RuntimeError(f"Erro Inicializando bllEmbeddings: {e}")
        
def get_bllEmbeddings(session=Session):
    if bllEmbeddings is None:
        initBllEmbeddings( session)
    return bllEmbeddings


class EmbeddingsBll:
    def __init__(self):
        from main import localconfig as localcfg
        self.localconfig = localcfg
        self.max_length = int(self.localconfig.get("max_length"))        
        if self.max_length > 1024:
            raise RuntimeError("max_length não pode ser maior que 1024.")        
        self.logger = logger.log        
        self.tokenizer = None        
        self.embedding_dim = None               
        # ===== PyTorch GPU (para indexação em lote) =====
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_model = None  # type: ignore                

    
    # Código antigo que estava quebrando o texto com 1024 caracteres
    # # Gera embeddings para uma lista de textos agilizando assim a indexação
    # # Gera embeddings em lote usando PyTorch + GPU (batch dinâmico)
    # #Se len(texts) não for múltiplo de 16 → padding com textos vazios.
    # #Retorna lista de embeddings na ordem original.   
    # def generate_embeddings(self, texts: List[str], ids: Optional[List[Optional[int]]] = None):                
    #     if not texts:
    #         return []

    #     # ===== limpeza e truncamento (igual ao que você já fazia) =====
    #     clean_texts = []
    #     for txt in texts:
    #         if not txt or not isinstance(txt, str):
    #             clean_texts.append("")
    #         else:
    #             clean = ''.join(c for c in txt if ord(c) >= 32 and ord(c) != 127)
    #             clean_texts.append(clean.strip()[:self.max_length])

    #     # ===== tokenização (CPU) =====
    #     inputs = self.tokenizer(
    #         clean_texts,
    #         return_tensors="pt",
    #         truncation=True,
    #         padding=True,                 # padding dinâmico → melhor desempenho
    #         max_length=self.max_length,
    #     ) # type: ignore

    #     # ===== move para GPU (async) =====
    #     inputs = {
    #         k: v.to(self.device, non_blocking=True)
    #         for k, v in inputs.items()
    #     }

    #     # ===== forward GPU =====
    #     with torch.no_grad():
    #         embeddings = self.torch_model(
    #             input_ids=inputs["input_ids"],
    #             attention_mask=inputs["attention_mask"],
    #         ) # type: ignore

    #     # ===== volta para CPU / numpy =====
    #     embeddings = embeddings.detach().cpu().numpy().astype("float32")

    #     return embeddings

    def _forward_embeddings(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.torch_model is None:
            raise RuntimeError("torch_model não carregado. Chame load_model_and_tokenizer() antes.")

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    emb = self.torch_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
            else:
                emb = self.torch_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

        return emb


    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        ids: Optional[Union[int, List[Optional[int]]]] = None
    ) -> Union[Optional[np.ndarray], List[Optional[np.ndarray]]]:

        # ---- modo single/batch ----
        if isinstance(texts, str):
            texts_list = [texts]
            single_mode = True
        else:
            texts_list = texts
            single_mode = False

        # ✅ checagem aqui (ANTES de tokenizar)
        if self.tokenizer is None:
            raise RuntimeError("tokenizer não carregado. Chame load_model_and_tokenizer() antes.")            

        # ---- normaliza ids (só para logs/erros) ----
        if ids is not None:
            if isinstance(ids, int):
                ids_list: List[Optional[int]] = [ids]
            else:
                ids_list = ids
            if len(ids_list) != len(texts_list):
                raise ValueError("O número de IDs deve corresponder ao número de texts.")
        else:
            ids_list = [None] * len(texts_list)

        # ---- limpeza (SEM truncar por caracteres) ----
        clean_texts: List[Optional[str]] = [None] * len(texts_list)
        valid_indices: List[int] = []

        for idx, text in enumerate(texts_list):
            try:
                if not text or not isinstance(text, str):
                    continue

                clean_text = ''.join(c for c in text if ord(c) >= 32 and ord(c) != 127).strip()

                if not clean_text:
                    continue

                clean_texts[idx] = clean_text
                valid_indices.append(idx)

            except Exception as e:
                raise RuntimeError(f"Erro ao limpar texto para ID {ids_list[idx]}: {e}")

        if not valid_indices:
            return None if single_mode else [None] * len(texts_list) # type: ignore

        # ---- tokenização (truncamento por TOKENS aqui) ----
        batch_texts = [clean_texts[i] for i in valid_indices]  # type: ignore

        try:
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,  # <-- truncamento correto por tokens
                padding=True,
                pad_to_multiple_of=8 if self.device.type == "cuda" else None,
            ) # type: ignore

            # move tensores para o device correto
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            emb = self._forward_embeddings(inputs)

            embeddings = emb.detach().cpu().numpy().astype("float32")  # (batch, dim)        

            # ---- remonta na ordem original ----
            results: List[Optional[np.ndarray]] = [None] * len(texts_list)
            for j, orig_idx in enumerate(valid_indices):
                results[orig_idx] = embeddings[j].reshape(-1)

            return results[0] if single_mode else results

        except Exception as e:
            error_ids = [ids_list[i] for i in valid_indices]
            error_texts = [clean_texts[i] for i in valid_indices]
            raise RuntimeError(f"Erro ao gerar embeddings para IDs {error_ids} e texts {error_texts}: {e}")


    #Gera embedding para um texto
    def generate_embedding(self, text: str, Id: Optional[int]) -> Optional[np.ndarray]:        
        result = self.generate_embeddings(texts=text, ids=Id)
        return result  # type: ignore
    
    #Carrega o modelo e tokenizer."""
    def load_model_and_tokenizer(self) -> None:       
        try:
            model_path  = f"{self.localconfig.getModelPath()}-dense-onnx"
            onnx_path_batch = f"{model_path}/bge-m3-dense.onnx"
            print_with_time(f"[INFO] Carregando tokenizer e modelo de: {onnx_path_batch}")                        

            # 1. Carrega o tokenizer (essencial!)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )

            # ===== Carrega modelo PyTorch para indexação em lote (GPU) =====
            # OBS: aqui usamos o modelo HF original (não ONNX).
            hf_model_path = self.localconfig.getModelPath()

            base_model = AutoModel.from_pretrained(
                hf_model_path,
                local_files_only=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )

            self.torch_model =  dense_embedding_wrapperModule.DenseEmbeddingTorchWrapper(base_model).to(self.device).eval()

            # TF32 ajuda bastante em GPUs modernas (mantém precisão boa pra embeddings)
            if self.device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True                      
            
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar tokenizer ou modelo: {e}")