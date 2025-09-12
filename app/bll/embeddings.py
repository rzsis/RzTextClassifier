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
        IdEncontrado: int
        CodClasse: int
        Classe: str
        Similaridade: float


    # Classe para representar médias por classe
    class ClasseMedia(BaseModel):
        CodClasse: int
        Classe: str
        Media: float

    # Classe para representar quantidades por classe
    class ClasseQtd(BaseModel):
        CodClasse: int
        Classe: str
        Quantidade: int

    # Classe principal que inclui os campos e a lista de itens encontrados
    class ResultadoSimilaridade(BaseModel):
        IdEncontrado: int
        CodClasse: int
        Classe: str
        Similaridade: float
        Metodo: str
        CodClasseMedia: int
        CodClasseQtd: int
        ListaSimilaridade: List['EmbenddingsModule.ItemSimilar']
        ListaClassesMedia: List['EmbenddingsModule.ClasseMedia']
        ListaClassesQtd: List['EmbenddingsModule.ClasseQtd']


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


    def search_similarities(self, query_embedding: np.ndarray, top_k: int = 5) -> 'EmbenddingsModule.ResultadoSimilaridade':
        """
        Realiza busca de similaridade usando o índice FAISS pré-criado e retorna resultados formatados.

        Args:
            query_embedding (np.ndarray): Embedding normalizado do texto de consulta.
            top_k (int): Número de resultados mais similares a retornar (padrão: 5).

        Returns:
            EmbenddingsModule.ResultadoSimilaridade: Objeto contendo o item pai, a lista de itens encontrados,
            médias por classe e quantidades por classe.

        Raises:
            RuntimeError: Se houver erro ao realizar a busca.
        """
        try:
            faiss.normalize_L2(query_embedding)
            distances, indices = self.index.search(query_embedding, top_k)
            results = [
                {
                    "Id": self.metadata["Id"][idx],
                    "Similaridade": float(dist),
                    "Classe": self.metadata["Classe"][idx],
                    "CodClasse": int(self.metadata["CodClasse"][idx])
                }
                for dist, idx in zip(distances[0], indices[0]) if idx != -1
            ]
            if not results:
                raise RuntimeError("Nenhum resultado encontrado na busca de similaridade.")

            # Dicionários para calcular média de similaridade e contagem por classe
            medias_por_classe = defaultdict(list)  # Lista de similaridades por CodClasse
            contagem_por_classe = defaultdict(int)  # Contagem de itens com similaridade > 80%

            # Processar resultados para calcular médias e contagens
            for result in results:
                if result["Similaridade"] > 0.80:
                    medias_por_classe[result["CodClasse"]].append(result["Similaridade"])
                    contagem_por_classe[result["CodClasse"]] += 1

            # Calcular média por classe
            medias = {cod_classe: sum(sims) / len(sims) if sims else 0.0 
                      for cod_classe, sims in medias_por_classe.items()}

            # Encontrar a classe com maior média
            classe_maior_media = max(medias.items(), key=lambda x: x[1], default=(None, 0.0))[0]
            media_maior = medias.get(classe_maior_media, 0.0)

            # Encontrar a classe com maior número de ocorrências
            classe_maior_qtd = max(contagem_por_classe.items(), key=lambda x: x[1], default=(None, 0))[0]
            qtd_maior = contagem_por_classe.get(classe_maior_qtd, 0)

            # Criar listas de médias e quantidades por classe
            lista_classes_media = [
                self.ClasseMedia(
                    CodClasse=cod_classe,
                    Classe=self.metadata["Classe"][self.metadata["CodClasse"] == cod_classe][0],
                    Media=media
                )
                for cod_classe, media in medias.items() if media > 0
            ]
            lista_classes_qtd = [
                self.ClasseQtd(
                    CodClasse=cod_classe,
                    Classe=self.metadata["Classe"][self.metadata["CodClasse"] == cod_classe][0],
                    Quantidade=quantidade
                )
                for cod_classe, quantidade in contagem_por_classe.items() if quantidade > 0
            ]

            # Criar lista de ItemEncontrado
            lista_itens = []
            for result in results:
                metodo = "E" if result["Similaridade"] > 0.97 else "M"
               # cod_classe_media = classe_maior_media if classe_maior_media is not None else result["CodClasse"]
               # cod_classe_qtd = classe_maior_qtd if classe_maior_qtd is not None else result["CodClasse"]

                # Verificar se a classe com maior média tem menos de 2 itens
                if media_maior > 0 and contagem_por_classe.get(classe_maior_media, 0) < 2:
                    if qtd_maior > 4:  # Usar a classe com mais ocorrências se tiver mais de 4
                        metodo = "Q"
                        cod_classe_media = classe_maior_qtd

                lista_itens.append(
                    self.ItemSimilar(
                        IdEncontrado=result["Id"],
                        CodClasse=result["CodClasse"],
                        Classe=result["Classe"],
                        Similaridade=result["Similaridade"]     
                    )
                )

            # Determinar o item pai (primeiro item se similaridade > 97%, senão o de maior similaridade)
            item_pai = results[0] if results[0]["Similaridade"] > 0.97 else max(results, key=lambda x: x["Similaridade"])
            lista_final = [item for item in lista_itens if item.IdEncontrado != item_pai["Id"]]

            # Retornar objeto ResultadoSimilaridade
            return self.ResultadoSimilaridade(
                IdEncontrado=item_pai["Id"],
                CodClasse=item_pai["CodClasse"],
                Classe=item_pai["Classe"],
                Similaridade=item_pai["Similaridade"],
                Metodo="E" if item_pai["Similaridade"] > 0.97 else "M",
                CodClasseMedia=classe_maior_media if classe_maior_media is not None else item_pai["CodClasse"],
                CodClasseQtd=classe_maior_qtd if classe_maior_qtd is not None else item_pai["CodClasse"],
                ListaSimilaridade=lista_final,
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