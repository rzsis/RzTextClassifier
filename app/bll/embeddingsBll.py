# embeddingsBll.py
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from db_utils import Session
import gpu_utils as gpu_utilsModule
from typing import  List, Optional,Union
from common import print_with_time
import logger
from qdrant_utils import Qdrant_Utils as qdrant_utilsModule

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
        initBllEmbeddings(session)
    return bllEmbeddings


class EmbeddingsBll:
    def __init__(self):
        from main import localconfig as localcfg
        self.localconfig = localcfg
        self.max_length = self.localconfig.get("max_length")
        self.max_txt_lenght = self.max_length * 8
        self.logger = logger.log
        self.gpu_utils = gpu_utilsModule.GpuUtils()        
        self.qdrant_client = qdrant_utilsModule().get_client()
        self.tokenizer = None
        self.model = None


    #Gera embeddings para uma lista de textos agilizando assim a indexação
    def generate_embeddings(self, texts: Union[str, List[str]], ids: Optional[Union[int, List[int]]] = None) -> Union[Optional[np.ndarray], List[Optional[np.ndarray]]]:
    
        if isinstance(texts, str):
            texts = [texts]
            single_mode = True
        else:
            single_mode = False
        
        if ids is not None:
            if isinstance(ids, int):
                ids = [ids]
            if len(ids) != len(texts):
                raise ValueError("O número de IDs deve corresponder ao número de texts.")
        else:
            ids = [None] * len(texts) # type: ignore
        
        clean_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            try:
                if not text or not isinstance(text, str):
                    clean_texts.append(None)
                    continue
                
                clean_text = ''.join(c for c in text if ord(c) >= 32 and ord(c) != 127)
                clean_text = clean_text[0:self.max_txt_lenght].strip()

                if not clean_text:
                    clean_texts.append(None)
                    continue
                
                clean_texts.append(clean_text)
                valid_indices.append(idx)
            except Exception as e:
                raise RuntimeError(f"Erro ao limpar texto para ID {ids[idx]}: {e}") # type: ignore
        
        if not valid_indices:
            return None if single_mode else [None] * len(texts) # type: ignore
        
        try:
            inputs = self.tokenizer(
                [clean_texts[i] for i in valid_indices],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.model.device) # type: ignore
            
            with torch.no_grad():
                outputs = self.model(**inputs) # type: ignore
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            del inputs, outputs
            
            embeddings = embeddings.astype('float32')
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.divide(embeddings, norms, where=norms > 0)
            
            # Preenche os resultados na ordem original
            results = [None] * len(texts)
            for j, idx in enumerate(valid_indices):
                results[idx] = embeddings[j].flatten()
            
            return results[0] if single_mode else results # type: ignore
        
        except Exception as e:
            error_ids = [ids[i] for i in valid_indices if i in valid_indices] # type: ignore
            error_texts = [clean_texts[i] for i in valid_indices if i in valid_indices]
            raise RuntimeError(f"Erro ao gerar embeddings para IDs {error_ids} e texts {error_texts}: {e}")
    
    #Gera embedding para um texto
    def generate_embedding(self, text: str, Id: Optional[int]) -> np.ndarray:        
        clean_text = ""
        try:
            if not text or not isinstance(text, str):
                return None # pyright: ignore[reportReturnType]
            
            clean_text = ''.join(c for c in text if ord(c) >= 32 and ord(c) != 127)
            clean_text = clean_text[0:self.max_txt_lenght].strip()

            if not clean_text:
                return None # pyright: ignore[reportReturnType]
            
            inputs = self.tokenizer(
                clean_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.model.device) # pyright: ignore[reportOptionalCall]

            with torch.no_grad():
                outputs = self.model(**inputs) # pyright: ignore[reportOptionalCall]
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            del inputs, outputs
            
            embedding = embedding.astype('float32')
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.flatten()
        except Exception as e:
            raise RuntimeError(f"Erro ao gerar embedding para texto ID = {Id} -> {clean_text} : {e}")
        
    #Carrega o modelo e tokenizer."""
    def load_model_and_tokenizer(self) -> None:       
        try:
            model_path = self.localconfig.getModelPath()
            if not os.path.exists(model_path):
                raise RuntimeError(f"Diretório do modelo {model_path} não encontrado.")
            
            use_gpu = torch.cuda.is_available()

            self.gpu_utils.clear_gpu_cache()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            self.model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if use_gpu else torch.float32,
                    trust_remote_code=True,
                    device_map="auto"  # GPU se existir, CPU se não
            ).eval()

            device_load = ""
            if next(self.model.parameters()).is_cuda:
               device_load = "✅ Modelo e tokenizer carregados na GPU"
            else:
               device_load = "🧠 Modelo e tokenizer carregados na CPU"


            print_with_time(f"{device_load} de {model_path} com dimensão {self.model.config.hidden_size}")
            

            if self.model.config.hidden_size > 1024:
                raise RuntimeError(f"Dimensão do embedding maior que 1024 não suportado.")
                        

        except Exception as e:
            raise RuntimeError(f"Erro ao carregar tokenizer ou modelo: {e}")
