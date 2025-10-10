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
            self.collection_name = f"v{localcfg.get('codcli')}_textos_classificar"
            self.k = 20
            self.similarity_threshold = 0.95
            self.min_similars = 3
            self.clusters = {} # Cache: {id_base: [{"id": id_similar, "score": score}, ...]}
            # Inicializa embeddings
            embeddingsBllModule.initBllEmbeddings(self.session)
            self.qdrant_utils = Qdrant_UtilsModule()
            self.qdrant_client = self.qdrant_utils.get_client()
            self.qdrant_utils.create_collection(self.collection_name)
            self.classifica_textoBll = classifica_textoBllModule(embeddingsModule=embeddingsBllModule.bllEmbeddings, session=session)
            self.log_ClassificacaoBll = LogClassificacaoBllModule(session)
            self.logger = logger.log
            self.baseWhereSQLClassificar = """
                                    WHERE Indexado = false
                                    and Classificado = true
                                    and t.TxtTreinamento IS NOT NULL and t.TxtTreinamento <> ''
                                    and t.Metodo in ('N','Q','M')                                    
                                """
            
            self.baseWhereSQLBuscarSimilar = """
                                    WHERE Indexado = true
                                    and Classificado = true
                                    and t.TxtTreinamento IS NOT NULL and t.TxtTreinamento <> ''      
                                    and t.BuscouSimilar = false                
                                    and t.Metodo in ('N','Q','M')                                    
                                """   
            self.gpu_utils = gpu_utilsModule.GpuUtils()
            self.limiteItensClassificar = 5000
                
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
        

    #obtem a quantidade de textos pendentes
    def _get_Textos_falta_buscar_similar(self) -> Sequence[RowMapping]:
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
        
    #processa os textos que faltam buscar similares e faz uma pesquisa no qdrant para encontrar textos similares
    def processa_textos_falta_buscar_similar(self):
        try:
            print_with_time(f"Iniciando busca de textos similares...")
            data = self._get_Textos_falta_buscar_similar()
            if not data:
                sucessMessage = "Nenhum texto textos similar restante para classifica."
                print_with_time(sucessMessage)
                return {
                    "status": "OK",
                    "processados": sucessMessage,
                    "restante": f"Restam 0 textos pendentes."
                }
            
            similares_inseridos = 0
            for row in tqdm(data, desc="Processando textos para busca de similares"):
                try:
                    embedding   = embeddingsBllModule.bllEmbeddings.generate_embedding(row['Text'],row['id'])
                    similars    = self._search_qdrant(embedding, row['id'])
                    if len(similars) >= self.min_similars:
                        self._insere_sugestao_textos_classificar(row['id'], similars)                        
                        similares_inseridos += 1
                
                except Exception as e:
                    self.logger.error(f"Erro ao processar texto id {row['id']}: {e}")

            self._mark_as_buscou_similar(data)                    
            
            sucessMessage = f"Inseridos {similares_inseridos} sugestões de textos similares."
            print_with_time(sucessMessage)
            self.gpu_utils.clear_gpu_cache()
            return {
                "status": "OK",
                "processados": sucessMessage,
                "restante": f"Restam {self._get_qtd_textos_falta_buscar_similar()} textos pendentes."
            }
        except Exception as e:
            errorMessage = f"Erro ao processar textos para busca de similares: {e}"
            print_error(errorMessage)
            return {
                "status": "ERROR",
                "processados": errorMessage,
                "restante": ""
            }
        
    #insere as sugestões de textos similares na tabela sugestao_textos_classificar
    def _insere_sugestao_textos_classificar(self, id_texto: int, similars: list[dict]):
        try:
            query = f"""
                select * from sugestao_textos_classificar 
                where IdBase = :id_texto or IdSimilar = :id_textoSimilar
            """
            sugestos_existentes = self.session.execute(
                    text(query),
                    {
                        "id_texto": id_texto,
                        "id_textoSimilar": id_texto,                        
                    }
            ).mappings().all()

             
            if len(sugestos_existentes) > 0:                
                return
            
            for similar in similars:
                try:
                    query = """
                        INSERT INTO sugestao_textos_classificar (IdBase, IdSimilar, Similaridade, DataHora)
                        VALUES (:id_base, :id_similar, :similaridade, NOW())
                    """
                    self.session.execute(
                        text(query),
                        {
                            "id_base": id_texto,
                            "id_similar": similar['id'],
                            "similaridade": (similar['score'] or 0)*100
                        }
                    )
                    self.session.commit()
                except Exception as e:
                    self.logger.error(f"Erro ao inserir sugestao_textos_classificar para id {id_texto}: {e}")
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


    #obtem a quantidade de textos pendentes
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

    #obtem os dados a serem indexados
    def _fetch_data_to_classify(self) -> Sequence[RowMapping]:
        try:
            query = f"""
                SELECT t.id, t.TxtTreinamento AS Text
                FROM textos_classificar t
                {self.baseWhereSQLClassificar}
                ORDER BY t.id
                LIMIT {self.limiteItensClassificar}
            """
            return self.session.execute(text(query)).mappings().all()
        except Exception as e:
            raise RuntimeError(f"Erro ao obter dados do banco em textos_classificar: {e}")

    #depois de processado marca como indexado
    def _mark_as_indexado(self, id_texto: int):
        try:
            query = """
                UPDATE textos_classificar
                SET indexado = true
                WHERE id = :id_texto
            """
            self.session.execute(text(query), {"id_texto": id_texto})
            self.session.commit()
        except Exception as e:
            self.logger.error(f"Erro ao marcar texto como indexado (id: {id_texto}): {e}")
            self.session.rollback()

    def _mark_lista_as_indexado(self, processados: list[dict]):
        try:
            # Filtra apenas os registros com UpInsertOk=True
            ids_to_update = [item['Id'] for item in processados if item['UpInsertOk']]
            if not ids_to_update:
               print_with_time("Nenhum texto para marcar como indexado.")
               return

            query = """
                UPDATE textos_classificar
                SET indexado = true
                WHERE id IN :ids
            """
            self.session.execute(text(query), {"ids": tuple(ids_to_update)})
            self.session.commit()
            self.logger.info(f"Marcados {len(ids_to_update)} textos como indexados em lote.")
        except Exception as e:
            self.logger.error(f"Erro ao marcar textos como indexados em lote: {e}")
            self.session.rollback()

    def _insert_texto_qdrant(self, id_texto: int, embedding: np.ndarray):
        try:           
            point = PointStruct(id=id_texto, vector=embedding.flatten().tolist(), payload={"id": id_texto})
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
        except Exception as e:
            self.logger.error(f"Erro ao inserir vetor no Qdrant para id {id_texto}: {e}")
            raise

    def _insert_lista_texto_qdrant(self, processed_data: list[dict]) -> list[dict]:
        try:
            points = []
            for item in tqdm(processed_data, desc="Inserindo dados no Qdrant"):
                embedding = item['Embedding']
                points.append(PointStruct(
                    id=item['Id'],
                    vector=embedding.flatten().tolist(),
                    payload={"id": item['Id']}
                ))
            
            result = self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            # Verifica se a operação foi bem-sucedida
            if result.status == "completed":
                for item in processed_data:
                    item['UpInsertOk'] = True
                self.logger.info(f"Inseridos {len(processed_data)} textos no Qdrant com sucesso.")
            else:
                for item in processed_data:
                    item['UpInsertOk'] = False
                self.logger.error(f"Falha ao inserir textos no Qdrant: status {result.status}")
            
            return processed_data
        except Exception as e:
            self.logger.error(f"Erro ao inserir lista de textos no Qdrant: {e}")
            for item in processed_data:
                item['UpInsertOk'] = False
            return processed_data

    def _search_qdrant(self, embedding: np.ndarray, id_texto: int) -> list[dict]:
        try:
            RuntimeError("Trocar para buscar_embedding de qdrant_utils")
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=embedding.flatten().tolist(),
                limit=self.k,
                score_threshold=self.similarity_threshold,
                query_filter=Filter(
                    must_not=[FieldCondition(key="id", match=MatchValue(value=id_texto))]
                )
            )
            high_similars = [
                {"id": int(res.payload["id"]), "score": res.score} # pyright: ignore[reportOptionalSubscript]
                for res in search_results
                if int(res.payload["id"]) != id_texto # pyright: ignore[reportOptionalSubscript]
            ]
            return high_similars
        except Exception as e:
            self.logger.error(f"Erro ao buscar similares no Qdrant para id {id_texto}: {e}")
            return []


    #pega todos os textos pendentes que não foram processados no qdrant, gera o embedding e insere no qdrant
    def indexa_e_classifica_textos_classificar(self) -> dict:
        print_with_time(f"Iniciando indexação e detecção de textos similares...")
        self.clusters = {} # Reseta cache
        data = self._fetch_data_to_classify()
        if not data:
            sucessMessage = "Nenhum texto pendente para processar."
            print_with_time(sucessMessage)
            return {
                "status": "OK",
                "processados": sucessMessage,
                "restante": f"Restam 0 textos pendentes."
            }

        # Acumula embeddings em uma lista de dicionários com Id, Embedding e UpInsertOk
        processed_data = []
        for i, row in enumerate(tqdm(data, desc="Gerando embeddings")):
            try:
                embedding = embeddingsBllModule.bllEmbeddings.generate_embedding(row['Text'],row['id'])
                # Clear cache every X batches
                if i % 20 == 0:
                     self.gpu_utils.clear_gpu_cache()

                processed_data.append({
                    'Id': row['id'],
                    'Embedding': embedding,
                    'UpInsertOk': False  # Inicialmente False, será atualizado após upsert
                })
            except Exception as e:
                self.logger.error(f"Erro ao gerar embedding para id {row['id']}: {e}")
                               
        self.gpu_utils.clear_gpu_cache()

        # Insere no Qdrant em lotes menores e atualiza UpInsertOk
        insert_qDrant_Batch_Size = 200  # Define o tamanho do lote
        for i in tqdm(range(0, len(processed_data), insert_qDrant_Batch_Size), desc="Processando lotes no Qdrant"):
            batch_data = processed_data[i:i + insert_qDrant_Batch_Size]
            batch_data = self._insert_lista_texto_qdrant(batch_data)
            # Marca como indexado apenas os textos com UpInsertOk=True no lote atual
            self._mark_lista_as_indexado(batch_data)

        self.processa_textos_falta_buscar_similar()
        self.gpu_utils.clear_gpu_cache()

        sucessMessage = f"Processados {len([item for item in processed_data if item['UpInsertOk']])} textos pendentes com Qdrant."
        print_with_time(sucessMessage)
        return {
            "status": "OK",
            "processados": sucessMessage,
            "restante": f"Restam {self._get_qtd_textos_pendentes_classificar()} textos pendentes."
        }
    