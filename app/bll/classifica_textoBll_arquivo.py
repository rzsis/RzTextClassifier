#classifica_textoBll.py
import re
import string
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from requests import session
from sqlalchemy import text
import bll.log_ClassificacaoBll as log_ClassificacaoBllModule
from bll.embeddingsBll import EmbeddingsBll  # Importing the original module
from collections import defaultdict
import faiss
from transformers import AutoTokenizer
from sqlalchemy.orm import Session

class classifica_textoBll:
    def __init__(self, embeddingsModule: EmbeddingsBll, session: Session):        
        self.embeddingsModule = embeddingsModule          
        self.log_ClassificacaoBll = log_ClassificacaoBllModule.LogClassificacaoBll(session)
        self.session = session  
        

    # Pydantic model classes
    class ItemSimilar(BaseModel):
        IdEncontrado: Optional[int]
        CodClasse: Optional[int]
        Classe: Optional[str]
        Similaridade: Optional[float]

    class ClassesInfo(BaseModel):
        CodClasse: Optional[int]
        Classe: Optional[str]
        Quantidade: Optional[int]        
        Media: Optional[float]

    class ResultadoSimilaridade(BaseModel):
        IdEncontrado: Optional[int]
        CodClasse: Optional[int]
        Classe: Optional[str]
        Similaridade: Optional[float]
        Metodo: Optional[str]
        CodClasseMedia: Optional[int]
        CodClasseQtd: Optional[int]
        ListaSimilaridade: Optional[List['classifica_textoBll.ItemSimilar']]
        ListaClassesInfo: Optional[List['classifica_textoBll.ClassesInfo']]        
    

    #obtem o id com maior similaridade para a codclasse
    def _get_best_id_by_codclasse(self, results: List[dict], cod_classe: int) -> Optional[int]:
        max_sim_item = max([result for result in results 
                                    if result["CodClasse"] == cod_classe],
                                    key=lambda x: x["Similaridade"],
                                    default=None
                            )  

        return max_sim_item["IdEncontrado"] if max_sim_item else None                 

    def search_similarities(self, query_embedding: np.ndarray, id_a_classificar:Optional[int] = None, 
                                TabelaOrigem:Optional[str] = "", 
                                top_k: int = 20,
                                gravar_log = False) -> 'classifica_textoBll.ResultadoSimilaridade':
        min_similarity = 0.8
        try:
            faiss.normalize_L2(query_embedding)
            
            distances, indices = self.embeddingsModule.index.search(query_embedding, top_k) # pyright: ignore[reportCallIssue]
            results = [
                {
                    "IdEncontrado": self.embeddingsModule.metadata["Id"][idx],
                    "Similaridade": float(dist),
                    "Classe": self.embeddingsModule.metadata["Classe"][idx],
                    "CodClasse": int(self.embeddingsModule.metadata["CodClasse"][idx])
                }
                for dist, idx in zip(distances[0], indices[0]) if idx != -1 and float(dist) > min_similarity
            ]
            if not results:
                return self.ResultadoSimilaridade(
                    IdEncontrado=None,
                    CodClasse=None,
                    Classe=f"Não encontrada similaridade superior {(min_similarity or 0 )*100}%",
                    Similaridade=None,
                    Metodo="N",
                    CodClasseMedia=None,
                    CodClasseQtd=None,
                    ListaSimilaridade=None,
                    ListaClassesInfo=None
                )

            # Create mapping of CodClasse to Classe
            classe_map = dict(zip(self.embeddingsModule.metadata["CodClasse"], self.embeddingsModule.metadata["Classe"]))

            # Process results
            medias_por_classe = defaultdict(list)
            contagem_por_classe = defaultdict(int)
            max_sim_por_classe = defaultdict(lambda: None)  # Store item directly
            metodo = ""
            item_pai = None

            for result in results:
                if result["Similaridade"] >= 0.97 and metodo != "E":
                    metodo = "E"                    
                    item_pai = result
                    

                cod_classe = result["CodClasse"]
                medias_por_classe[cod_classe].append(result["Similaridade"])
                contagem_por_classe[cod_classe] += 1

                if max_sim_por_classe[cod_classe] is None or result["Similaridade"] > max_sim_por_classe[cod_classe]["Similaridade"]: # pyright: ignore[reportOptionalSubscript]
                    max_sim_por_classe[cod_classe] = result

            # Calculate averages
            medias = {cod_classe: sum(sims) / len(sims) for cod_classe, sims in medias_por_classe.items()}
            classe_maior_media = max(medias.items(), key=lambda x: x[1], default=(None, 0.0))[0]            

            # Get class with the highest count
            classe_maior_qtd = max(contagem_por_classe.items(), key=lambda x: x[1], default=(None, 0))[0]

            # Create ClassesInfo list combining media and quantidade
            lista_classes_info = [
                self.ClassesInfo(
                    CodClasse=cod_classe,
                    Classe=classe_map.get(cod_classe, "Nenhuma"),
                    Media=medias.get(cod_classe, 0.0),
                    Quantidade=contagem_por_classe.get(cod_classe, 0)
                )
                for cod_classe in set(list(medias.keys()) + list(contagem_por_classe.keys()))
                if medias.get(cod_classe, 0) > 0 or contagem_por_classe.get(cod_classe, 0) > 0
            ] or [self.ClassesInfo(CodClasse=None, Classe="Nenhuma", Media=0.0, Quantidade=0)]

            maior_item_media    = max(lista_classes_info,key=lambda x: x.Media or 0)
            maior_item_qtd      = max(lista_classes_info,key=lambda x: x.Quantidade or 0)
            # Determine method and parent item if not "E"
            if metodo != "E":
                if  (maior_item_media.Media >= 0.91) and (maior_item_media.Quantidade >= 3):  # pyright: ignore[reportOptionalOperand]
                    # Find the item in results with the highest Similaridade for the CodClasse with the highest Media                                     
                    metodo = "M"
                    item_pai = {
                        "IdEncontrado": self._get_best_id_by_codclasse(results, maior_item_media.CodClasse),
                        "CodClasse": maior_item_media.CodClasse,
                        "Classe" : maior_item_media.Classe,
                        "Similaridade": maior_item_media.Media
                    }
                elif (maior_item_qtd.Quantidade >= 4) and (maior_item_qtd.Media >= 0.87): # pyright: ignore[reportOptionalOperand]
                    # Find the item in results with the highest Qtd for the CodClasse with the highest Qtd                
                    metodo = "Q"
                    item_pai = {
                        "IdEncontrado":self._get_best_id_by_codclasse(results, maior_item_qtd.CodClasse),
                        "CodClasse": maior_item_qtd.CodClasse,
                        "Classe" : maior_item_qtd.Classe,
                        "Similaridade": maior_item_qtd.Media
                    }
                else:
                    metodo = "N"
                    item_pai = {
                        "IdEncontrado": None,
                        "CodClasse": None,
                        "Classe" : "",
                        "Similaridade": None
                    }
            if gravar_log:
                self.log_ClassificacaoBll.gravaLogClassificacao(item_pai["IdEncontrado"],  # pyright: ignore[reportOptionalSubscript]
                                                                id_a_classificar, 
                                                                metodo, 
                                                                TabelaOrigem,
                                                                item_pai["CodClasse"]) # pyright: ignore[reportOptionalSubscript]

            return self.ResultadoSimilaridade(
                IdEncontrado=item_pai["IdEncontrado"], # pyright: ignore[reportOptionalSubscript]
                CodClasse=item_pai["CodClasse"], # pyright: ignore[reportOptionalSubscript]
                Classe=item_pai["Classe"], # pyright: ignore[reportOptionalSubscript]
                Similaridade=item_pai["Similaridade"], # pyright: ignore[reportOptionalSubscript]
                Metodo=metodo,
                CodClasseMedia=classe_maior_media,
                CodClasseQtd=classe_maior_qtd,
                ListaSimilaridade=[self.ItemSimilar(**result) for result in results],
                ListaClassesInfo=lista_classes_info
            )
        except Exception as e:
            raise RuntimeError(f"Erro ao buscar similaridades: {e}")

    ###   Classifica um texto com base na similaridade com embeddings de referência.        
    def classifica_texto(self, texto: str, id_a_classificar: Optional[int] = None,
                        TabelaOrigem:Optional[str] = "",
                        top_k: int = 20,
                        gravar_log = False) -> 'classifica_textoBll.ResultadoSimilaridade':
        
        try:
            # Generate embedding for the input text to compare in future
            query_embedding = self.embeddingsModule.generate_embedding(texto,id_a_classificar)
            
           
            # Perform similarity search
            return self.search_similarities(query_embedding,
                                            id_a_classificar , 
                                            TabelaOrigem, 
                                            top_k,
                                            gravar_log)
        
        except Exception as e:
            raise RuntimeError(f"Erro ao classificar texto: {e}")