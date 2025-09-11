#embeddings.py
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
    def load_model_and_embendings(self, embedding_type: str):              

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

        ids = self.metadata["Id"]
        Classes = self.metadata["Classe"]
        CodClasse = self.metadata["CodClasse"].astype(int)
    

        #Verificar consistência entre embeddings e metadados
        if len(self.embeddings) != len(ids) or len(self.embeddings) != len(CodClasse) or len(self.embeddings) != len(Classes):
            raise RuntimeError(f"Erro: Inconsistência entre embeddings ({len(self.embeddings)}) e metadados "
                            f"(ids: {len(ids)}, CodClasse: {len(CodClasse)}, Texts: {len(Classes)})")                 
        
        # Normalizar embeddings de referência e criar índice FAISS usando normalize_embeddings
        self.embeddings = self.normalize_embeddings(self.embeddings, "embeddings de referência")
        print_with_time(f"Índice FAISS criado com {self.index.ntotal} vetores.")    
        return self.embeddings, self.metadata    


    def search_similarities(self, query_embedding: np.ndarray, top_k: int = 5) -> list:
        """
        Realiza busca de similaridade usando o índice FAISS pré-criado.

        Args:
            query_embedding (np.ndarray): Embedding normalizado do texto de consulta.
            top_k (int): Número de resultados mais similares a retornar (padrão: 5).

        Returns:
            list: Lista de dicionários contendo 'Id', 'similaridade' e 'Classe' dos resultados mais similares.

        Raises:
            RuntimeError: Se houver erro ao realizar a busca.
        """
        try:
            faiss.normalize_L2(query_embedding)
            distances, indices = self.index.search(query_embedding, top_k)
            results = [
                {
                    "Id": self.metadata["Id"][idx],
                    "similaridade": float(dist),
                    "Classe": self.metadata["Classe"][idx]
                }
                for dist, idx in zip(distances[0], indices[0]) if idx != -1
            ]
            return results
        except Exception as e:
            raise RuntimeError(f"Erro ao buscar similaridades: {e}")
        

    #normaliza os embeddings para poder comparar
    def normalize_embeddings(self, embeddings: np.ndarray, context: str = "embeddings") -> np.ndarray:
        """
        Normaliza os embeddings fornecidos, cria um índice FAISS e o armazena em self.index.
        
        Args:
            embeddings (np.ndarray): Array de embeddings a serem normalizados.
            context (str): Descrição do contexto para mensagens de erro (ex: 'embeddings de referência').
        
        Returns:
            np.ndarray: Embeddings normalizados.
        
        Raises:
            RuntimeError: Se houver valores inválidos (NaN/Inf) ou falha na criação do índice.
        """
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
            self.index.add(embeddings)

            # Verificar se todos os vetores foram adicionados ao índice
            if self.index.ntotal != len(embeddings):
                raise RuntimeError(f"Erro: {self.index.ntotal} vetores adicionados ao índice, esperado {len(embeddings)}")
            
            return embeddings
        
        except Exception as e:
            raise RuntimeError(f"Erro ao normalizar embeddings para {context}: {e}")             
        
    ##Classifica um texto com base na similaridade com embeddings de referência.
    def classifica_texto(self, texto: str, top_k: int = 5) -> list:
        """
        Classifica um texto com base na similaridade com embeddings de referência.

        Args:
            texto (str): Texto a ser classificado.
            top_k (int): Número de resultados mais similares a retornar (padrão: 5).

        Returns:
            list: Lista de dicionários contendo 'Id', 'similaridade' e 'Classe' dos resultados mais similares.

        Raises:
            RuntimeError: Se houver erro ao gerar o embedding ou realizar a busca.
        """
        try:
            # Gerar embedding para o texto de consulta
            query_embedding = self.generate_embedding(texto)
            
            # Normalizar o embedding de consulta
            query_embedding = query_embedding.astype('float32')
            
            # Realizar a busca de similaridades
            return self.search_similarities(query_embedding, top_k)
        
        except Exception as e:
            raise RuntimeError(f"Erro ao classificar texto: {e}")