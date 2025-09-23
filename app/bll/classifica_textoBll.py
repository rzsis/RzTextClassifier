# classifica_texto_bll.py
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

    class ClasseMedia(BaseModel):
        CodClasse: Optional[int]
        Classe: Optional[str]
        Media: Optional[float]

    class ClasseQtd(BaseModel):
        CodClasse: Optional[int]
        Classe: Optional[str]
        Quantidade: Optional[int]

    class ResultadoSimilaridade(BaseModel):
        IdEncontrado: Optional[int]
        CodClasse: Optional[int]
        Classe: Optional[str]
        Similaridade: Optional[float]
        Metodo: Optional[str]
        CodClasseMedia: Optional[int]
        CodClasseQtd: Optional[int]
        ListaSimilaridade: Optional[List['classifica_textoBll.ItemSimilar']]
        ListaClassesMedia: Optional[List['classifica_textoBll.ClasseMedia']]
        ListaClassesQtd: Optional[List['classifica_textoBll.ClasseQtd']]


    def search_similarities(self, query_embedding: np.ndarray, top_k: int = 20) -> 'classifica_textoBll.ResultadoSimilaridade':
        """
        Searches for similar embeddings in the reference set.

        Args:
            query_embedding (np.ndarray): The query embedding to search with.
            top_k (int): Number of top similar results to return (default: 20).

        Returns:
            ResultadoSimilaridade: Object containing similarity results, including lists of similar items,
                                  class averages, and class counts.

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
                    ListaClassesMedia=None,
                    ListaClassesQtd=None
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

            # Create lists using a helper function
            def create_class_list(classe_map, data, model_class, field_name):
                return [
                    model_class(
                        CodClasse=cod_classe,
                        Classe=classe_map.get(cod_classe, "Nenhuma"),
                        **{field_name: value}
                    )
                    for cod_classe, value in data.items() if value > 0
                ] or [model_class(CodClasse=None, Classe="Nenhuma", **{field_name: 0})]

            lista_classes_media = create_class_list(classe_map, medias, self.ClasseMedia, "Media")
            lista_classes_qtd = create_class_list(classe_map, contagem_por_classe, self.ClasseQtd, "Quantidade")

            # Determine method and parent item if not "E"
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
                ListaSimilaridade=[self.ItemSimilar(**result) for result in results],
                ListaClassesMedia=lista_classes_media,
                ListaClassesQtd=lista_classes_qtd
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
            ResultadoSimilaridade: Object containing similarity results, including lists of similar items,
                                  class averages, and class counts.

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