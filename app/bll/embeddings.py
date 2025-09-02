import os
import json
from pathlib import Path
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from datetime import datetime
from common import print_with_time
from collections import defaultdict
import csv
import localconfig


class Embenddings:
    def __init__(self,localcfg:localconfig):
        self.localconfig = localcfg
        

    # Função para gerar embedding para comparação do texto
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

    def load_model_and_embendings(self):      
        print_with_time(f"Inicializando modelos e embeddings...")
        # Carregar tokenizer e modelo para gerar novos embeddings caso necessario
        try:
            model_path = self.localconfig.getModelPath()
            if (os.path.exists(model_path) == False):
                raise RuntimeError(f"Diretório do modelo {model_path} não encontrado.")
                exit(1)                 

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
            self.model.eval()
            self.max_length = self.localconfig.read_config().get("max_length")

            print_with_time(f"Modelo e tokenizer carregados de {model_path}")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar tokenizer ou modelo: {e}")
            exit(1)


        # Carregar embeddings e metadados
        embeddings_file = Path(self.localconfig.getEmbendingPath(),f"train_final_text.npy")
        metadata_file = Path(self.localconfig.getEmbendingPath(),f"train_final_metadata.npz")

        if not embeddings_file.exists() or not metadata_file.exists():
            raise RuntimeError(f"Arquivos de embeddings {embeddings_file.stem} ou metadados {metadata_file.stem} não encontrados em {self.localconfig.getEmbendingPath()}")

        try:            
            embeddings = np.load(embeddings_file)            
            metadata = np.load(metadata_file, allow_pickle=True)
            print_with_time(f"Embeddings de referência: {embeddings_file} e metadados: {metadata_file} carregados com sucesso.")            
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar embeddings ou metadados para {embeddings_file.stem}: {e}")
            exit(1)

    #normaliza os embeddings para poder comparar
    def normalize_embeddings(self, embeddings: np.ndarray, texto:str) -> np.ndarray:
        try:
            # Verificar se há NaN ou Inf nos embeddings
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                raise RuntimeError(f"Erro: Embedding inválido (NaN ou Inf) para {texto}")
                    
            #Normalizar embedding de consulta
            embeddings = embeddings.astype('float32')
            faiss.normalize_L2(embeddings)
            if  torch.cuda.is_available():        
                torch.cuda.empty_cache()                        
        except Exception as e:
            raise RuntimeError(f"Erro: Embedding inválido (NaN ou Inf) para {texto} {e}")                        
        

    def classifica_texto(self, texto: str):
        try:
            generate_embedding = self.generate_embedding(texto)
            self.normalize_embeddings(generate_embedding,texto)    



            return generate_embedding.tolist()
        
        except Exception as e:
            raise RuntimeError(f"Erro ao classificar texto: {e}")