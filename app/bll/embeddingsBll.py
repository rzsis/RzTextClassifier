#embeddingsBll.py
import os
import json
from pathlib import Path
from click import Option
import numpy as np
import faiss
from shtab import Optional
from sympy import N
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from datetime import datetime
from bll import ids_e_classes_corretasBll
from common import print_with_time
from collections import defaultdict
import csv
from db_utils import Session
import localconfig
from pydantic import BaseModel
from typing import List, Optional
import bll.ids_e_classes_corretasBll  as IdsEClassesCorretasBllModule

bllEmbeddings = None

def initBllEmbeddings(session=Session):
    global bllEmbeddings    
    if bllEmbeddings is None:
        try:
            from main import localconfig  # importa localconfig do main.py                
            bllEmbeddings = EmbeddingsBll()  # inicializa modelos (carrega embeddings)
            bllEmbeddings.load_model_and_embendings("train",session)  # carrega os embeddings finais        
        except Exception as e:
            raise RuntimeError(f"Erro Inicializando bllEmbeddings: {e}")            

        

class EmbeddingsBll:
    def __init__(self):
        from main import localconfig as localcfg  # importa localconfig do main.py            
        self.localconfig = localcfg
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Makes errors immediate
        os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enables device-side assertions        

    # Função para gerar embedding para comparação do texto, transformando o texto em um vetor numérico
    def generate_embedding(self, text: str) -> np.ndarray:
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Liberar tensores intermediários
            del inputs, outputs
            return embedding
        except Exception as e:
            raise RuntimeError(f"Erro ao gerar embedding: {e}")            

    #Inicializa os modelos e carrega os embeddings
    #embedding_type pode ser train ou final
    def load_model_and_embendings(self, embedding_type: str,session: Session) -> tuple[np.ndarray, np.ndarray]:              

        # define os caminhos dos arquivos de embeddings e metadados
        if embedding_type == "train":
            embeddings_file = Path(self.localconfig.getEmbeddingsTrain(),f"train_text.npy")
            metadata_file = Path(self.localconfig.getEmbeddingsTrain(),f"train_metadata.npz")
        elif embedding_type == "final":
            embeddings_file = Path(self.localconfig.getEmbendingFinal(),f"train_final_text.npy")
            metadata_file = Path(self.localconfig.getEmbendingFinal(),f"train_final_metadata.npz")
        else:
            raise RuntimeError("embedding_type deve ser 'train' ou 'final'")

        if not embeddings_file.exists() or not metadata_file.exists():
            raise RuntimeError(f"Arquivos de embeddings {embeddings_file.stem} ou metadados {metadata_file.stem} não encontrados em {self.localconfig.getEmbendingFinal()}")
            
        print_with_time(f"Inicializando modelos e embeddings...")
        # Carregar tokenizer e modelo para gerar novos embeddings caso necessario
        try:
            model_path = self.localconfig.getModelPath()
            if (os.path.exists(model_path) == False):
                raise RuntimeError(f"Diretório do modelo {model_path} não encontrado.")                                 

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
            self.model.eval()
            self.max_length = self.localconfig.read_config().get("max_length")
            print_with_time(f"Modelo e tokenizer carregados de {model_path}")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar tokenizer ou modelo: {e}")            

        try:            
            self.embeddings = np.load(embeddings_file)            
            self.metadata = np.load(metadata_file, allow_pickle=True)
            print_with_time(f"Embeddings de referência: {embeddings_file} e metadados: {metadata_file} carregados com sucesso.")            
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar embeddings ou metadados para {embeddings_file.stem}: {e}")            

        # Convert NpzFile to dict to improve perfomance
        self.metadata = {key: self.metadata[key] for key in self.metadata}

        ids = self.metadata["Id"]
        Classes = self.metadata["Classe"]
        CodClasse = self.metadata["CodClasse"].astype(int)
    

        #Verificar consistência entre embeddings e metadados
        if len(self.embeddings) != len(ids) or len(self.embeddings) != len(CodClasse) or len(self.embeddings) != len(Classes):
            raise RuntimeError(f"Erro: Inconsistência entre embeddings ({len(self.embeddings)}) e metadados "
                            f"(ids: {len(ids)}, CodClasse: {len(CodClasse)}, Texts: {len(Classes)})")                 


        idsEClassesCorretasBll = IdsEClassesCorretasBllModule.IdsEClassesCorretasBll(session)  # inicializa idsEClassesCorretasBll
        self.metadata = idsEClassesCorretasBll.corrige_metadata(self.metadata)


        # Normalizar embeddings de referência e criar índice FAISS usando normalize_embeddings
        self.embeddings = self.normalize_embeddings(self.embeddings, "embeddings de referência")

        print_with_time(f"Índice FAISS criado com {self.index.ntotal} vetores.")    

        return self.embeddings, self.metadata     # pyright: ignore[reportReturnType]


                
    #normaliza os embeddings para poder comparar
    def normalize_embeddings(self, embeddings: np.ndarray, context: str = "embeddings") -> np.ndarray:        
        try:
            # Verificar se há NaN ou Inf nos embeddings
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                raise RuntimeError(f"Erro: Embedding inválido (NaN ou Inf) para {context}")
                    
            # Converter para float32 e normalizar embeddings
            embeddings = embeddings.astype('float32')
            faiss.normalize_L2(embeddings)
            
            # Limpar cache da GPU, se disponível e necessário
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Criar índice FAISS (mantido na CPU)
            self.dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)  # Inner Product para cosine similarity
            self.index.add(embeddings) # pyright: ignore[reportCallIssue]

            # Verificar se todos os vetores foram adicionados ao índice
            if self.index.ntotal != len(embeddings):
                raise RuntimeError(f"Erro: {self.index.ntotal} vetores adicionados ao índice, esperado {len(embeddings)}")
            
            return embeddings
        
        except Exception as e:
            raise RuntimeError(f"Erro ao normalizar embeddings para {context}: {e}")             
        