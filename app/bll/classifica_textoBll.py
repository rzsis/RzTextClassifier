from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from bll.embeddings import EmbeddingsModule  # Importing the original module
from collections import defaultdict
import faiss
from transformers import AutoTokenizer

class classifica_textoBll:
    def __init__(self, embeddingsModule: EmbeddingsModule):
        # Initialize the EmbenddingsModule with the provided localconfig
        self.embeddingsModule = embeddingsModule        

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

    def search_similarities(self, query_embedding: np.ndarray, top_k: int = 20) -> 'classifica_textoBll.ResultadoSimilaridade':
        """
        Searches for similar embeddings in the reference set.

        Args:
            query_embedding (np.ndarray): The query embedding to search with.
            top_k (int): Number of top similar results to return (default: 20).

        Returns:
            ResultadoSimilaridade: Object containing similarity results, including lists of similar items
                                  and class information (average and count).

        Raises:
            RuntimeError: If the search process fails.
        """
        min_similarity = 0.8
        try:
            faiss.normalize_L2(query_embedding)
            distances, indices = self.embeddingsModule.index.search(query_embedding, top_k)
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
                    Classe=f"Não encontrada similaridade superior {min_similarity*100}%",
                    Similaridade=None,
                    Metodo=None,
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

                if max_sim_por_classe[cod_classe] is None or result["Similaridade"] > max_sim_por_classe[cod_classe]["Similaridade"]:
                    max_sim_por_classe[cod_classe] = result

            # Calculate averages
            medias = {cod_classe: sum(sims) / len(sims) for cod_classe, sims in medias_por_classe.items()}
            classe_maior_media = max(medias.items(), key=lambda x: x[1], default=(None, 0.0))[0]
            media_maior = medias.get(classe_maior_media, 0.0)

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

            maior_item_media = max(lista_classes_info,key=lambda x: x.Media or 0)
            maior_item_qtd = max(lista_classes_info,key=lambda x: x.Quantidade or 0)
            # Determine method and parent item if not "E"
            if metodo != "E":
                if (maior_item_media.Media >= 0.91) and (maior_item_media.Quantidade >= 3):
                    # Find the item in results with the highest Similaridade for the CodClasse with the highest Media                
                    max_sim_item = max([result for result in results 
                            if result["CodClasse"] == maior_item_media.CodClasse],
                            key=lambda x: x["Similaridade"],
                            default=None
                    )                    
                    metodo = "M"
                    item_pai = {
                        "IdEncontrado": max_sim_item["IdEncontrado"] if max_sim_item else None,
                        "CodClasse": maior_item_media.CodClasse,
                        "Classe" : maior_item_media.Classe,
                        "Similaridade": maior_item_media.Media
                    }
                elif (maior_item_qtd.Quantidade >= 4) and (maior_item_qtd.Media >= 0.87):
                    # Find the item in results with the highest Qtd for the CodClasse with the highest Qtd                
                    max_sim_item = max([result for result in results 
                            if result["CodClasse"] == maior_item_qtd.CodClasse],
                            key=lambda x: x["Quantidade"],
                            default=None
                    )   

                    metodo = "Q"
                    item_pai = {
                        "IdEncontrado": max_sim_item["IdEncontrado"] if max_sim_item else None,
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

            return self.ResultadoSimilaridade(
                IdEncontrado=item_pai["IdEncontrado"],
                CodClasse=item_pai["CodClasse"],
                Classe=item_pai["Classe"],
                Similaridade=item_pai["Similaridade"],
                Metodo=metodo,
                CodClasseMedia=classe_maior_media,
                CodClasseQtd=classe_maior_qtd,
                ListaSimilaridade=[self.ItemSimilar(**result) for result in results],
                ListaClassesInfo=lista_classes_info
            )
        except Exception as e:
            raise RuntimeError(f"Erro ao buscar similaridades: {e}")

    def classifica_texto(self, texto: str, top_k: int = 20) -> 'classifica_textoBll.ResultadoSimilaridade':
        """
        Classifica um texto com base na similaridade com embeddings de referência.

        Args:
            texto (str): Texto a ser classificado.
            top_k (int): Número de resultados mais similares a retornar (padrão: 20).

        Returns:
            ResultadoSimilaridade: Object containing similarity results, including lists of similar items
                                  and class information (average and count).

        Raises:
            RuntimeError: If embedding generation or similarity search fails.
        """
        try:
            # Generate embedding for the input text to compare in future
            query_embedding = self.embeddingsModule.generate_embedding(texto)
            
            # Normalize the query embedding
            query_embedding = query_embedding.astype('float32')
            
            # Perform similarity search
            return self.search_similarities(query_embedding, top_k)
        
        except Exception as e:
            raise RuntimeError(f"Erro ao classificar texto: {e}")