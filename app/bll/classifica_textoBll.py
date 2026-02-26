#classifica_textoBll.py
import re
import string
from typing_extensions import runtime
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from requests import session
from sqlalchemy import text
import bll.log_ClassificacaoBll as log_ClassificacaoBllModule
from bll.embeddingsBll import EmbeddingsBll  # Importing the original module
from collections import defaultdict
from transformers import AutoTokenizer
from sqlalchemy.orm import Session
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule
from dbClasses.classes_utils import get_ClassesUtils as dbClasses_utils_get_ClassesUtils


class classifica_textoBll:
    def __init__(self, embeddingsModule: EmbeddingsBll, session: Session):             
        self.embeddingsModule = embeddingsModule          
        self.log_ClassificacaoBll = log_ClassificacaoBllModule.LogClassificacaoBll(session)
        self.session = session  
        self.qdrant_utils = Qdrant_UtilsModule()  # Initialize Qdrant_Utils instance       
        self.final_collection_name = self.qdrant_utils.get_collection_name("final")
        
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
        IdPesquisado: Optional[int] = None  # ID do texto que foi pesquisado , necessario para busca por Ids
    

    #obtem o id com maior similaridade para a codclasse
    def _get_best_id_by_codclasse(self, results: List[dict], cod_classe: int) -> Optional[int]:
        max_sim_item = max([result for result in results 
                                    if result["CodClasse"] == cod_classe],
                                    key=lambda x: x["Similaridade"],
                                    default=None
                            )  

        return max_sim_item["IdEncontrado"] if max_sim_item else None                 

    #obtem a lista de similaridades a partir do embedding de consulta e retorna um resultado estruturado
    #exclusion_list é uma lista de Ids que devem ser excluidos da busca de similaridade pois não faz sentido eu procurar os proprios ids como similares deles mesmos
    def get_similarity_list(self, 
                                query_embedding: np.ndarray, 
                                collection_name: str,
                                id_a_classificar:Optional[int] = None, 
                                TabelaOrigem:Optional[str] = "", 
                                itens_limit: int = 20,
                                gravar_log = False,
                                min_similarity:int = 0.8,
                                metodo_selecao: list[str] = ['E'],
                                return_ListaSimilares: bool = True) -> 'classifica_textoBll.ResultadoSimilaridade':                
        try:
            if 'E' not in metodo_selecao:
                raise RuntimeError(f"Método de seleção 'E' é obrigatório para garantir que a classificação seja feita por similaridade exata quando disponível. Métodos selecionados: {metodo_selecao}")
                
            results = self.qdrant_utils.search_embedding(embedding= query_embedding,
                                                         collection_name= collection_name,
                                                         itens_limit= itens_limit,
                                                         similarity_threshold= min_similarity
                                                         )
                    
                                                            
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

            classe_map = {r["CodClasse"]: r["Classe"] for r in results}
            
            # Process results
            medias_por_classe = defaultdict(list)
            contagem_por_classe = defaultdict(int)
            max_sim_por_classe = defaultdict(lambda: None)  # Store item directly
            metodo_classificacao_encontrado = ""
            item_pai = {
                        "IdEncontrado": None,
                        "CodClasse": None,
                        "Classe" : None,
                        "Similaridade": None
                    }
            classe_maior_media = None
            classe_maior_qtd = None
            lista_classes_info = None


            if 'E' in metodo_selecao:
                for result in results:
                    if result["Similaridade"] >= 0.97 and metodo_classificacao_encontrado != "E":
                        metodo_classificacao_encontrado = "E"                    
                        item_pai = result
                        if not any(m in metodo_selecao for m in ['M','Q']): # pyright: ignore[reportOptionalOperand]
                            break  # Se "E" for encontrado e "M" ou "Q" não estiverem na lista, pode sair do loop
                        
                    cod_classe = result["CodClasse"]
                    medias_por_classe[cod_classe].append(result["Similaridade"])
                    contagem_por_classe[cod_classe] += 1

                    if max_sim_por_classe[cod_classe] is None or result["Similaridade"] > max_sim_por_classe[cod_classe]["Similaridade"]: # pyright: ignore[reportOptionalSubscript]
                        max_sim_por_classe[cod_classe] = result


            if (( 'M' in metodo_selecao ) or ( 'Q' in metodo_selecao )) and (metodo_classificacao_encontrado != "E"): # pyright: ignore[reportOptionalOperand]
                # Calcula médias
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
                if  (maior_item_media.Media >= 0.91) and (maior_item_media.Quantidade >= 3) and ('M' in metodo_selecao):  # pyright: ignore[reportOptionalOperand]
                    # Find the item in results with the highest Similaridade for the CodClasse with the highest Media                                     
                    metodo_classificacao_encontrado = "M"
                    item_pai = {
                        "IdEncontrado": self._get_best_id_by_codclasse(results, maior_item_media.CodClasse),
                        "CodClasse": maior_item_media.CodClasse,
                        "Similaridade": maior_item_media.Media
                    }
                elif (maior_item_qtd.Quantidade >= 4) and (maior_item_qtd.Media >= 0.87) and ('Q' in metodo_selecao): # pyright: ignore[reportOptionalOperand]
                    # Find the item in results with the highest Qtd for the CodClasse with the highest Qtd                
                    metodo_classificacao_encontrado = "Q"
                    item_pai = {
                        "IdEncontrado":self._get_best_id_by_codclasse(results, maior_item_qtd.CodClasse),
                        "CodClasse": maior_item_qtd.CodClasse,                        
                        "Similaridade": maior_item_qtd.Media
                    }
                else:
                    metodo_classificacao_encontrado = "N"
                    item_pai = {
                        "IdEncontrado": None,
                        "CodClasse": None,
                        "Classe" : "",
                        "Similaridade": None
                    }
            elif metodo_classificacao_encontrado != "E":
                metodo_classificacao_encontrado = "N"
                
            if gravar_log:
                self.log_ClassificacaoBll.gravaLogClassificacao(item_pai["IdEncontrado"],  # pyright: ignore[reportOptionalSubscript]
                                                                id_a_classificar, 
                                                                metodo_classificacao_encontrado, 
                                                                TabelaOrigem,
                                                                item_pai["CodClasse"]) # pyright: ignore[reportOptionalSubscript]

            if return_ListaSimilares:
                result_similaridade = [self.ItemSimilar(**result) for result in results]
            else:
                result_similaridade = None  

            if (item_pai != None) and (item_pai["IdEncontrado"] != None) and (item_pai["CodClasse"] != None):
                item_pai["Classe"] = dbClasses_utils_get_ClassesUtils().get_nome_classe(item_pai["CodClasse"]) or None



            return self.ResultadoSimilaridade(
                IdEncontrado=item_pai["IdEncontrado"], # pyright: ignore[reportOptionalSubscript]
                CodClasse=item_pai["CodClasse"], # pyright: ignore[reportOptionalSubscript]
                Classe=item_pai["Classe"], # pyright: ignore[reportOptionalSubscript]
                Similaridade=item_pai["Similaridade"], # pyright: ignore[reportOptionalSubscript]
                Metodo=metodo_classificacao_encontrado,
                CodClasseMedia=classe_maior_media,
                CodClasseQtd=classe_maior_qtd,
                ListaSimilaridade=result_similaridade,
                ListaClassesInfo=lista_classes_info
            )
        except Exception as e:
            raise RuntimeError(f"Erro ao buscar similaridades: {e}")

    ###   Classifica um texto com base na similaridade com embeddings de referência.        
    def classifica_texto(self, 
                         texto: str, 
                         id_a_classificar: Optional[int] = None,
                         TabelaOrigem:Optional[str] = "",
                         limite_itens: int = 20,
                         gravar_log = False) -> 'classifica_textoBll.ResultadoSimilaridade':
        
        try:
            # Generate embedding for the input text to compare in future
            query_embedding = self.embeddingsModule.generate_embedding(texto,id_a_classificar)
                       
            return self.get_similarity_list(query_embedding=query_embedding,                                            
                                            collection_name=self.final_collection_name,
                                            id_a_classificar=id_a_classificar , 
                                            TabelaOrigem=TabelaOrigem, 
                                            itens_limit=limite_itens,
                                            gravar_log=gravar_log
                                            )        
        except Exception as e:
            raise RuntimeError(f"Erro em classificar texto: {e}")
        
    # Retorna os textos a classificar a partir de uma lista de ids,
    def _get_textos_classificar_by_ids(self, lista_ids: List[int]) -> dict:
        try:
            if not lista_ids:
                return {}
            
            placeholders = ", ".join([":id" + str(i) for i in range(len(lista_ids))])
            query = f"SELECT id, TxtTreinamento FROM textos_classificar WHERE id IN ({placeholders})"
            params = {f"id{i}": id for i, id in enumerate(lista_ids)}
            result = self.session.execute(text(query), params).mappings().all()
            return {row["id"]: row["TxtTreinamento"] for row in result}
        except Exception as e:
            raise RuntimeError(f"Erro ao buscar textos por IDs: {e}")


    # classifica uma lista de ids e retorna a similaridade de cada um com os textos de referência, o resultado é uma lista de ResultadoSimilaridade para cada id classificado
    def classifica_ids(self, 
                         lista_ids: List[int],                                                                            
                         gravar_log = True,
                         metodo_selecao: List[str] = ['E']) -> List['classifica_textoBll.ResultadoSimilaridade']:
        
        if (any(metodo not in ['E','M','Q'] for metodo in metodo_selecao)):
            raise ValueError(f"Métodos de seleção {metodo_selecao} inválidos. Aceitos: 'E', 'M', 'Q'.")   


        lista_textos = self._get_textos_classificar_by_ids(lista_ids)
        
        resultados = []
        for id, texto in lista_textos.items():
            try:
                query_embedding = self.embeddingsModule.generate_embedding(texto,id)

                resultado_similaridade = self.get_similarity_list(query_embedding=query_embedding,                                            
                                            collection_name=self.final_collection_name,                                                                                        
                                            itens_limit=30,
                                            gravar_log=gravar_log,
                                            return_ListaSimilares=False,
                                            )                  
                if (resultado_similaridade.IdEncontrado != None):   
                    resultado_similaridade.IdPesquisado = id
                    resultados.append(resultado_similaridade)

            except Exception as e:
                raise RuntimeError(f"Erro ao classificar ID {id}: {e}")

        
        return resultados