#check_collidingBLL.py
import numpy as np
from requests import session
from sqlalchemy.orm import Session
from bll.classifica_textoBll import classifica_textoBll as classifica_textoBllModule
from bll.embeddingsBll import EmbeddingsBll
from common import print_with_time
from qdrant_utils import Qdrant_Utils as QdrantUtilsModule

class check_collidingBLL:
    def __init__(self, session: Session): 
        from main import localconfig as localcfg
        self.session = session
        self.localcfg = localcfg
        self.embeddings_handler = EmbeddingsBll()        
        self.qdrant_utils = QdrantUtilsModule()
        self.final_collection = self.qdrant_utils.get_collection_name("final")
        self.min_similarity = 0.97

    # varre a lista de similares e retorna os que colidem que tem classe diferente
    def _get_colliding_items(self,idBase:int,codClasse:int, lista_similares:dict[str,any]) -> dict[str,any]:
        try:
            ids_Colidentes_Atuais = []
            for similar_item in lista_similares:            
                neighbor_id         = similar_item["IdEncontrado"]
                neighbor_cod_classe = similar_item["CodClasse"]                
                if neighbor_id == idBase:
                    continue

                # Caso a classe for diferente, é colidência
                if (codClasse != neighbor_cod_classe):                    # type: ignore                            
                    if not (neighbor_id in ids_Colidentes_Atuais): # type: ignore
                        similar_item["Similaridade"] = similar_item["Similaridade"] * 100  # type: ignore
                        ids_Colidentes_Atuais.append(similar_item)  # type: ignore                  
                
            return ids_Colidentes_Atuais # type: ignore
        except Exception as e:
            raise RuntimeError(f"Erro em _get_colliding_items: {e}")
    

    #verfica colidencia baseado em data , data deve vir de qdrant_uitls.get_id esse é o formato esperado
    def check_colliding_by_Embedding(self, query_embedding:  np.array, id_found:int, CodClasse:int) -> dict[str,any]:        
        # Verifica se é um ndarray
        if not isinstance(query_embedding, np.ndarray):
            raise TypeError("O parâmetro 'query_embedding' deve ser um numpy.ndarray")

        # Verifica se não está vazio
        if query_embedding.size == 0:
            raise ValueError("O parâmetro 'query_embedding' está vazio")

        # Verifica se é numérico
        if not np.issubdtype(query_embedding.dtype, np.number):
            raise TypeError("O parâmetro 'query_embedding' deve conter apenas valores numéricos")
                                                 
        try:
            similarity_list = self.qdrant_utils.search_embedding(
                embedding=query_embedding,
                collection_name=self.final_collection,
                limite_itens=20,
                similarity_threshold=self.min_similarity                                          
                )

        except Exception as e:
            raise RuntimeError(f"Erro em check_colliding_by_embedding: {e}")
                          
        return self._get_colliding_items(id_found,CodClasse, similarity_list)

        