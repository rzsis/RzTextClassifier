from ast import Dict, List
import os
from pathlib import Path
from typing_extensions import runtime
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, Filter, FieldCondition, MatchValue
from sqlalchemy import RowMapping, Sequence, text
from tqdm import tqdm
from sqlalchemy.orm import Session
from common import print_with_time, print_error, get_localconfig
from bll.classifica_textoBll import classifica_textoBll as classifica_textoBllModule
import bll.embeddingsBll as embeddingsBllModule
from bll.log_ClassificacaoBll import LogClassificacaoBll as LogClassificacaoBllModule
import gpu_utils as gpu_utilsModule
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule
import logger
from collections.abc import Sequence
import torch
from bll.indexa_textos_classificarBll import indexa_textos_classificarBll as indexa_textos_classificarBllModule
import time

class sugere_textos_classificarBll:
    def __init__(self, session: Session):
        """
        Inicializa a classe para indexação e detecção de textos similares usando Qdrant.
        Args:
            session (Session): Sessão SQLAlchemy para operações no banco.
        """
        try:
            from main import localconfig as localcfg
            self.session = session
            self.localconfig = localcfg
            self.config = localcfg.read_config()
            self.textos_classificar_collection_name = f"v{localcfg.get('codcli')}_textos_classificar"
            self.limite_similares = 20
            self.similarity_threshold = 0.95
            self.min_similars = 3
            self.clusters = {} # Cache: {id_base: [{"id": id_similar, "score": score}, ...]}
            # Inicializa embeddings
            embeddingsBllModule.initBllEmbeddings(self.session)
            self.qdrant_utils = Qdrant_UtilsModule()
            self.qdrant_client = self.qdrant_utils.get_client()
            self.qdrant_utils.create_collection(self.textos_classificar_collection_name)
            self.classifica_textoBll = classifica_textoBllModule(embeddingsModule=embeddingsBllModule.bllEmbeddings, session=session)
            self.log_ClassificacaoBll = LogClassificacaoBllModule(session)
            self.logger = logger.log
            LimitePalavras = localcfg.get("max_length")    
            self.baseWhereSQLBuscarSimilarCorreto = f"""
                                    WHERE 
                                        Indexado = true
                                    and Classificado = true
                                    and t.BuscouSimilar = false                                      
                                    and t.TxtTreinamento IS NOT NULL and t.TxtTreinamento <> ''                                                        
                                    and t.Metodo in ('N','Q','M') 
                                    and t.QtdPalavras <= {LimitePalavras}                                     
                                """   
                                         
            self.baseWhereSQLBuscarSimilar = f"""
                                    WHERE 
                                        Indexado = true                                    
                                    and t.BuscouSimilar = false                                      
                                    and t.TxtTreinamento IS NOT NULL and t.TxtTreinamento <> ''                                                                                            
                                    and t.QtdPalavras <= {LimitePalavras}                                     
                                """   
            self.gpu_utils = gpu_utilsModule.GpuUtils()
            self.limiteItensClassificar = localcfg.get("text_limit_per_batch")
                
        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar sugestao_textos_classificarBll: {e}")

    #obtem a quantidade de textos pendentes falta buscar similar
    def _get_qtd_textos_falta_buscar_similar(self) -> int:
        try:
            query = f"""
                SELECT Count(t.id) AS TotalTextosPendentes
                FROM textos_classificar t
                {self.baseWhereSQLBuscarSimilar}
                ORDER BY t.id
            """
            return self.session.execute(text(query)).mappings().all()[0]['TotalTextosPendentes']
        except Exception as e:
            raise RuntimeError(f"Erro ao obter _get_Textos_Pendentes: {e}")
        

    #Obtem a quantidade de textos pendentes
    def _get_textos_falta_buscar_similar(self) -> Sequence[RowMapping]:
        try:
            query = f"""
                SELECT t.id, t.TxtTreinamento AS Text
                FROM textos_classificar t
                {self.baseWhereSQLBuscarSimilar}
                ORDER BY t.id                
                LIMIT {self.limiteItensClassificar}                
            """
            return self.session.execute(text(query)).mappings().all()
        except Exception as e:
            raise RuntimeError(f"Erro ao obter _get_Textos_Pendentes: {e}")
        
    #faz a busca de similares e retorna a lista para inserir na sugestão de classificação        
    def get_similares(self,id:int) -> list: # type: ignore
        try:
            id_found        = self.qdrant_utils.get_id(id=id, collection_name=self.textos_classificar_collection_name)                    
            if (id_found == None):
                return None # type: ignore
                    
            result          = self.classifica_textoBll.search_similarities(query_embedding= id_found["Embedding"],
                                                                        collection_name=self.textos_classificar_collection_name,
                                                                        id_a_classificar= None,
                                                                        TabelaOrigem="C",
                                                                        itens_limit=50,
                                                                        gravar_log=False,
                                                                        min_similarity=self.similarity_threshold)
                    
            return [item.__dict__ for item in result.ListaSimilaridade] # type: ignore
        except Exception as e:
            print_with_time(f"erro em get_similares {e} ")
            
    #obtem uma lista de sugestao_textos_classificar gerando uma lista dupla com IdSimilar e IdBase igual
    #pois uma vez um IdInserido ele não deve ser considerado similar a outro logo não deve ser inserido novamente
    def get_list_sugestao_textos_classificar(self): # type: ignore
        try:
            query = f"""
                SELECT t.IdBase as Id
                FROM sugestao_textos_classificar t
                Group by t.IdBase                         
                Union
                SELECT t.IdSimilar as Id
                FROM sugestao_textos_classificar t
                Group by t.IdSimilar                                            
            """
            rows = self.session.execute(text(query)).mappings().all()
            return {row["Id"] for row in rows}

        except Exception as e:
            raise RuntimeError(f"Erro ao obter _get_Textos_Pendentes: {e}")
        

        
    #insere as sugestões de textos similares na tabela sugestao_textos_classificar
    def _insere_sugestao_textos_classificar(self, id_texto: int, similars: list[dict]):
        try:               
            for similar in similars:
                try:
                    query = """
                        INSERT ignore INTO  sugestao_textos_classificar (IdBase, IdSimilar, Similaridade, DataHora)
                        VALUES (:id_base, :id_similar, :similaridade, NOW())
                    """
                    self.session.execute(
                        text(query),
                        {
                            "id_base": id_texto,
                            "id_similar": similar['IdEncontrado'],
                            "similaridade": (similar['Similaridade'] or 0)*100
                        }
                    )
                    self.session.commit()
                except Exception as e:
                    print_with_time(f"Erro ao inserir sugestao_textos_classificar para id {id_texto}: {e}")
                    self.session.rollback()

        except Exception as e:
                self.logger.error(f"Erro ao inserir sugestões de textos similares para id {id_texto}: {e}")
                self.session.rollback()

    #depois de processado marca como BuscouSimilar a lista que foi processada
    def _mark_as_buscou_similar(self, data: Sequence[RowMapping]):
        try:
            ids_to_update = [row['id'] for row in data]
            query = """
                UPDATE textos_classificar
                SET BuscouSimilar = true
                WHERE id IN :ids
            """
            self.session.execute(text(query), {"ids": tuple(ids_to_update)})
            self.session.commit()
            self.logger.info(f"Marcados {len(ids_to_update)} textos como buscou similar em lote.")
        except Exception as e:
            self.logger.error(f"Erro ao marcar textos como buscou similar em lote: {e}")
            self.session.rollback()


    #Obtem a quantidade de textos pendentes a classificar
    def _get_qtd_textos_pendentes_classificar(self) -> int:
        try:
            query = f"""
                SELECT Count(t.id) AS TotalTextosPendentes
                FROM textos_classificar t
                {self.baseWhereSQLClassificar}
                ORDER BY t.id
            """
            return self.session.execute(text(query)).mappings().all()[0]['TotalTextosPendentes']
        except Exception as e:
            raise RuntimeError(f"Erro ao obter _get_Textos_Pendentes: {e}")

    #pega todos os textos pendentes que o sistema não buscou similar procura similares agrupa por similaridade
    #para depois o usuario sugerir classificações
    def sugere_textos_para_classificar(self) -> dict:
        try:
            #primeiro deve indexar tudo no qdrant para depois fazer a busca
            inicio = time.time()
            indexa_textos_classificarBll = indexa_textos_classificarBllModule(self.session)
            indexa_textos_classificarBll.indexa_textos_classificar()

            print_with_time(f"Iniciando busca de textos similares...")
            data = self._get_textos_falta_buscar_similar()
            sugestao_textos_classificar = self.get_list_sugestao_textos_classificar()

            if not data:
                sucessMessage = "Nenhum texto textos similar restante para classificar"
                print_with_time(sucessMessage)
                return {
                    "status": "OK",
                    "processados": sucessMessage,
                    "restante": f"0"
                }
            
            similares_inseridos = 0
            for row in tqdm(data, desc="Processando textos para busca de similares"):
                try:
                    lista_similares = self.get_similares(id=row['id'])                    
                    if (lista_similares != None):
                        already_exists  = False
                        for similar in lista_similares:                         
                            already_exists = (similar['IdEncontrado'] in sugestao_textos_classificar)
                            if already_exists:
                                break

                        if (already_exists == False) and len(lista_similares) >= self.min_similars:# caso tiver mais que X amostras de similares insere para sugerir para classificar                        
                            self._insere_sugestao_textos_classificar(row['id'], lista_similares)                        
                            similares_inseridos += len(lista_similares) 
                

                except Exception as e:
                    self.logger.error(f"Erro ao buscar similar de texto id {row['id']}: {e}")

            self._mark_as_buscou_similar(data)                    
            self.gpu_utils.clear_gpu_cache()

            tempo_decorrido_min = (time.time() - inicio) / 60          
            sucessMessage = f"Inseridos {similares_inseridos} sugestões de textos similares, Tempo decorrido: {tempo_decorrido_min:.2f} minutos"

            print_with_time(sucessMessage)
            return {
                "status": "OK",
                "mensagem": sucessMessage,
                "restante": f"{self._get_qtd_textos_falta_buscar_similar()}"
            }
        except Exception as e:
            errorMessage = f"Erro ao processar textos para busca de similares: {e}"
            print_error(errorMessage)
            return {
                "status": "ERROR",
                "processados": errorMessage,
                "restante": ""
            }
    