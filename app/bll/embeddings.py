#embeddings.py
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
from common import print_with_time
from collections import defaultdict
import csv
import localconfig
from pydantic import BaseModel
from typing import List, Optional

class EmbenddingsModule:
    def __init__(self,localcfg:localconfig):
        self.localconfig = localcfg
        
 # Classe para os itens da lista (cada item encontrado)
    class ItemSimilar(BaseModel):
        IdEncontrado: Optional[int]
        CodClasse: Optional[int]
        Classe: Optional[str]
        Similaridade: Optional[float]


    # Classe para representar médias por classe
    class ClasseMedia(BaseModel):
        CodClasse: Optional[int]
        Classe: Optional[str]
        Media: Optional[float]

    # Classe para representar quantidades por classe
    class ClasseQtd(BaseModel):
        CodClasse: Optional[int]
        Classe: Optional[str]
        Quantidade: Optional[int]

    # Classe principal que inclui os campos e a lista de itens encontrados
    class ResultadoSimilaridade(BaseModel):
        IdEncontrado: Optional[int]
        CodClasse: Optional[int]
        Classe: Optional[str]
        Similaridade: Optional[float]
        Metodo: Optional[str]
        CodClasseMedia: Optional[int]
        CodClasseQtd: Optional[int]
        ListaSimilaridade: Optional[List['EmbenddingsModule.ItemSimilar']]
        ListaClassesMedia: Optional[List['EmbenddingsModule.ClasseMedia']]
        ListaClassesQtd: Optional[List['EmbenddingsModule.ClasseQtd']]


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

    # Função auxiliar para criar listas de médias e quantidades para o retorno
    def create_class_list(self,classe_map,data, model_class, field_name):
                return [
                    model_class(
                        CodClasse=cod_classe,
                        Classe=classe_map.get(cod_classe, "Nenhuma"),
                        **{field_name: value}
                    )
                    for cod_classe, value in data.items() if value > 0
                ] or [model_class(CodClasse=None, Classe="Nenhuma", **{field_name: 0})]

    ##procura o texto mais similar no embeddings de referência
    def search_similarities(self, query_embedding: np.ndarray, top_k: int = 5) -> 'EmbenddingsModule.ResultadoSimilaridade':
        min_similarity = 0.8
        try:
            faiss.normalize_L2(query_embedding)
            distances, indices = self.index.search(query_embedding, top_k)
            results = [
                {
                    "IdEncontrado": self.metadata["Id"][idx],
                    "Similaridade": float(dist),
                    "Classe": self.metadata["Classe"][idx],
                    "CodClasse": int(self.metadata["CodClasse"][idx])
                }
                for dist, idx in zip(distances[0], indices[0]) if idx != -1 and float(dist) > min_similarity
            ]
            if not results:
                return self.ResultadoSimilaridade(
                    IdEncontrado=None,
                    CodClasse=None,
                    Classe=f"Não encontrada similaridade superior {min_similarity*100}%",
                    Similaridade=None,
                    Metodo=None,
                    CodClasseMedia=None,
                    CodClasseQtd=None,
                    ListaSimilaridade=None,
                    ListaClassesMedia=None,
                    ListaClassesQtd=None
                )

            # Criar mapeamento de CodClasse para Classe
            classe_map = dict(zip(self.metadata["CodClasse"], self.metadata["Classe"]))

            # Processar resultados
            medias_por_classe = defaultdict(list)
            contagem_por_classe = defaultdict(int)
            max_sim_por_classe = defaultdict(lambda: None)  # Armazenar item diretamente
            metodo = ""
            item_pai = None

            for result in results:
                if result["Similaridade"] >= 0.97 and metodo != "E":
                    metodo = "E"
                    item_pai = result
                cod_classe = result["CodClasse"]
                medias_por_classe[cod_classe].append(result["Similaridade"])
                contagem_por_classe[cod_classe] += 1
                if max_sim_por_classe[cod_classe] is None or result["Similaridade"] > max_sim_por_classe[cod_classe]["Similaridade"]:
                    max_sim_por_classe[cod_classe] = result

            # Calcular médias
            medias = {cod_classe: sum(sims) / len(sims) for cod_classe, sims in medias_por_classe.items()}
            classe_maior_media = max(medias.items(), key=lambda x: x[1], default=(None, 0.0))[0]
            media_maior = medias.get(classe_maior_media, 0.0)

            #obtem a classe com maior quantidade
            classe_maior_qtd = max(contagem_por_classe.items(), key=lambda x: x[1], default=(None, 0))[0]

            # Criar listas
            lista_classes_media = self.create_class_list(classe_map,medias, self.ClasseMedia, "Media")
            lista_classes_qtd = self.create_class_list(classe_map,contagem_por_classe, self.ClasseQtd, "Quantidade")

            # Determinar metodo e item_pai se não for "E"
            if metodo != "E":
                if media_maior >= 0.91:
                    metodo = "M"
                    item_pai = max_sim_por_classe[classe_maior_media]
                else:
                    metodo = "Q"
                    item_pai = max_sim_por_classe[classe_maior_qtd]

            return self.ResultadoSimilaridade(
                IdEncontrado=item_pai["IdEncontrado"],
                CodClasse=item_pai["CodClasse"],
                Classe=item_pai["Classe"],
                Similaridade=item_pai["Similaridade"],
                Metodo=metodo,
                CodClasseMedia=classe_maior_media,
                CodClasseQtd=classe_maior_qtd,
                ListaSimilaridade=results,
                ListaClassesMedia=lista_classes_media,
                ListaClassesQtd=lista_classes_qtd
            )
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