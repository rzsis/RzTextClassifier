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
        

    def load_model_and_embendings(self):      
        print_with_time(f"Inicializando modelos e embeddings...")
        # Carregar tokenizer e modelo para gerar novos embeddings caso necessario
        try:
            model_path = self.localconfig.getModelPath()
            if (os.path.exists(model_path) == False):
                raise RuntimeError(f"Diretório do modelo {model_path} não encontrado.")
                exit(1)                 
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
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

    def classifica_texto(self, strTexto: str):
        try:
            return 
        
        except Exception as e:
            raise RuntimeError(f"Erro ao classificar texto: {e}")