# embeddingsBll.py
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Makes errors immediate
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enables device-side assertions

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from db_utils import Session
import gpu_utils as gpu_utilsModule
from typing import Dict, List, Optional
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

    def load_model_and_tokenizer(self) -> None:
        """Carrega o modelo e tokenizer."""
        try:
            model_path = self.localconfig.getModelPath()
            if not os.path.exists(model_path):
                raise RuntimeError(f"Diretório do modelo {model_path} não encontrado.")
            
            self.gpu_utils.clear_gpu_cache()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                attn_implementation="eager"
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            device_load = ""
            if next(self.model.parameters()).is_cuda:
               device_load = "✅ Modelo e tokenizer carregados na GPU"
            else:
               device_load = "🧠 Modelo e tokenizer carregados na CPU"

            self.model.eval()
            print_with_time(f"{device_load} de {model_path} com dimensão {self.model.config.hidden_size}")
            

            if self.model.config.hidden_size > 1024:
                raise RuntimeError(f"Dimensão do embedding maior que 1024 não suportado.")
                        

        except Exception as e:
            raise RuntimeError(f"Erro ao carregar tokenizer ou modelo: {e}")
