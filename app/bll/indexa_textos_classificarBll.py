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
import time

class indexa_textos_classificarBll:
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
            self.baseWhereSQLNotIndexedCorreto = f"""
                                    WHERE Indexado = false
                                    and Classificado = true
                                    and t.TxtTreinamento IS NOT NULL and t.TxtTreinamento <> ''
                                    and t.Metodo in ('N','Q','M','') 
                                    and t.QtdPalavras <= {LimitePalavras}                                   
                                """                        
            
            self.baseWhereSQLNotIndexed = f"""                                     
                                    WHERE Indexado = false
                                    and t.TxtTreinamento IS NOT NULL and t.TxtTreinamento <> ''                                    
                                    and t.QtdPalavras <= {LimitePalavras}                                   
                                """                        
            self.gpu_utils = gpu_utilsModule.GpuUtils()
            self.limiteItensClassificar = localcfg.get("text_limit_per_batch")
                
        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar indexa_textos_classificarBll: {e}")

              
    #Obtem a quantidade de textos pendentes a classificar
    def _get_qtd_textos_pendentes_indexar(self) -> int:
        try:
            query = f"""
                SELECT Count(t.id) AS TotalTextosPendentes
                FROM textos_classificar t
                {self.baseWhereSQLNotIndexed}
                ORDER BY t.id
            """
            return self.session.execute(text(query)).mappings().all()[0]['TotalTextosPendentes']
        except Exception as e:
            raise RuntimeError(f"Erro ao obter _get_Textos_Pendentes: {e}")

    #obtem os dados a serem indexados que o sistema não tem na base de treinamento
    def _fetch_data_not_indexed(self) -> Sequence[RowMapping]:
        try:
            query = f"""
                SELECT t.id, t.TxtTreinamento AS Text
                FROM textos_classificar t
                {self.baseWhereSQLNotIndexed}
                ORDER BY t.id
                LIMIT {self.limiteItensClassificar}
            """
            return self.session.execute(text(query)).mappings().all()
        except Exception as e:
            raise RuntimeError(f"Erro ao obter dados do banco em textos_classificar: {e}")

    #marca os textos indexados no qdrant como indexados
    def _mark_lista_as_indexado(self, processados: list[dict]):
        try:
            # Filtra apenas os registros com UpInsertOk=True
            ids_to_update = [item['Id'] for item in processados if item['UpInsertOk']]
            if not ids_to_update:
               print_with_time("Nenhum texto para marcar como indexado.")
               return

            query = """
                UPDATE textos_classificar
                SET Indexado = true
                WHERE id IN :ids
            """
            self.session.execute(text(query), {"ids": tuple(ids_to_update)})
            self.session.commit()            
        except Exception as e:
            print_with_time(f"Erro ao marcar textos como indexados em lote: {e}")
            self.session.rollback()

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
                collection_name=self.textos_classificar_collection_name,
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
            print_with_time(f"Erro ao inserir lista de textos no Qdrant: {e}")
            for item in processed_data:
                item['UpInsertOk'] = False
            return processed_data

    #pega todos os textos pendentes que o sistema não conseguiu classifcar ou gerou médias ou quantidades
    #indexa no qdrant para buscar similares e definir uma busca mais precisa pro futuro
    def indexa_textos_classificar(self) -> dict:
        inicio = time.time()
        print_with_time(f"Iniciando indexação de textos a classificar...")
        self.clusters = {} # Reseta cache
        data = self._fetch_data_not_indexed()
        if not data:
            sucessMessage = "Nenhum texto pendente para processar."
            print_with_time(sucessMessage)
            return {
                "status": "OK",
                "mensagem": sucessMessage,
                "restante": f"0"
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
        
        self.gpu_utils.clear_gpu_cache()

        tempo_decorrido_min = (time.time() - inicio) / 60
        sucessMessage = f"Indexados {len([item for item in processed_data if item['UpInsertOk']])} textos pendentes no Qdrant, Tempo decorrido: {tempo_decorrido_min:.2f} minutos"
        print_with_time(sucessMessage)
        return {
            "status": "OK",
            "mensagem": sucessMessage,
            "restante": f"{self._get_qtd_textos_pendentes_indexar()}"
        }
    