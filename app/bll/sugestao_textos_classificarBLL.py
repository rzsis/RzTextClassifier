# sugestao_textos_classificarBll.py
import os
from pathlib import Path
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
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule
import logger
import faiss
from collections.abc import Sequence

class sugestao_textos_classificarBll:
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
            self.clusters = {}  # Cache: {id_base: [{"id": id_similar, "score": score}, ...]}
            # Inicializa embeddings
            embeddingsBllModule.initBllEmbeddings(self.session)
            self.embedding_dim = embeddingsBllModule.bllEmbeddings.dim


            self.qdrant_utils = Qdrant_UtilsModule()
            self.qdrant_client = self.qdrant_utils.get_client()            
            self.qdrant_utils.create_collection(self.collection_name)

            self.classifica_textoBll = classifica_textoBllModule(embeddingsModule=embeddingsBllModule.bllEmbeddings, session=session)
            self.log_ClassificacaoBll = LogClassificacaoBllModule(session)
            self.logger = logger.log
        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar sugestao_textos_classificarBll: {e}")

    def _get_Textos_Pendentes(self) -> Sequence[RowMapping]:
        try:
            query = """
                SELECT Count(t.id) AS TotalTextosPendentes
                FROM textos_classificar t
                WHERE Indexado = false
                ORDER BY t.id
            """
            return self.session.execute(text(query)).mappings().all()
        except Exception as e:
            raise RuntimeError(f"Erro ao obter _get_Textos_Pendentes: {e}")

    def _fetch_data(self) -> Sequence[RowMapping]:
        try:
            query = """
                SELECT t.id, t.TxtTreinamento AS Text
                FROM textos_classificar t
                WHERE Indexado = false
                ORDER BY t.id
                LIMIT 2000
            """
            return self.session.execute(text(query)).mappings().all()
        except Exception as e:
            raise RuntimeError(f"Erro ao obter dados do banco em textos_classificar: {e}")

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

    def _insert_qdrant(self, id_texto: int, embedding: np.ndarray):
        try:
            embedding = embedding.astype('float32')
            faiss.normalize_L2(embedding)
            point = PointStruct(id=str(id_texto), vector=embedding.flatten().tolist(), payload={"id": id_texto})
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            self.logger.info(f"Vetor inserido no Qdrant para id {id_texto}")
        except Exception as e:
            self.logger.error(f"Erro ao inserir vetor no Qdrant para id {id_texto}: {e}")
            raise

    def _search_qdrant(self, embedding: np.ndarray, id_texto: int) -> list[dict]:
        try:
            embedding = embedding.astype('float32')
            faiss.normalize_L2(embedding)
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

    def _check_existing_ids(self, high_similars: list[dict]) -> tuple[set, set]:
        try:
            ids = [similar["id"] for similar in high_similars]
            if not ids:
                return set(), set()
            query = """
                SELECT IdBase FROM sugestao_textos_classificar WHERE IdBase IN :ids
                UNION
                SELECT IdSimilar FROM sugestao_textos_classificar WHERE IdSimilar IN :ids
            """
            result = self.session.execute(text(query), {"ids": tuple(ids)}).fetchall()
            id_bases = set(row[0] for row in result if row[0] in [similar["id"] for similar in high_similars])
            id_similars = set(row[0] for row in result if row[0] not in id_bases)
            return id_bases, id_similars
        except Exception as e:
            self.logger.error(f"Erro ao checar IDs existentes: {e}")
            return set(), set()

    def _update_clusters(self, id_texto: int, high_similars: list[dict]):
        id_bases, id_similars = self._check_existing_ids(high_similars)
        if id_bases:
            max_score_id = max(
                (similar for similar in high_similars if similar["id"] in id_bases),
                key=lambda x: x["score"],
                default=None
            )
            if max_score_id and max_score_id["score"] >= self.similarity_threshold:
                id_base = max_score_id["id"]
                if id_base not in self.clusters:
                    self.clusters[id_base] = []
                self.clusters[id_base].append({"id": id_texto, "score": max_score_id["score"]})
                for similar in high_similars:
                    if similar["id"] != max_score_id["id"] and similar["id"] not in id_similars and similar["score"] >= self.similarity_threshold:
                        self.clusters[id_base].append({"id": similar["id"], "score": similar["score"]})
        else:
            if id_texto not in self.clusters:
                self.clusters[id_texto] = []
            for similar in high_similars:
                if similar["id"] not in id_similars and similar["score"] >= self.similarity_threshold:
                    self.clusters[id_texto].append({"id": similar["id"], "score": similar["score"]})

    def _insert_clusters_to_db(self):
        try:
            BATCH_SIZE = 100
            batch_params = []
            for id_base, similars in self.clusters.items():
                for similar in similars:
                    batch_params.append({
                        "id_base": id_base,
                        "id_similar": similar["id"],
                        "similaridade": similar["score"]
                    })

            if batch_params:
                query = """
                    INSERT INTO sugestao_textos_classificar (IdBase, IdSimilar, Similaridade, DataHora, CodClasse, JaClassificado)
                    VALUES (:id_base, :id_similar, :similaridade, NOW(), NULL, false)
                """
                for i in range(0, len(batch_params), BATCH_SIZE):
                    batch = batch_params[i:i + BATCH_SIZE]
                    self.session.execute(text(query), batch)
                    self.session.commit()
                    self.logger.info(f"Inseridas {len(batch)} sugestões em batch")
        except Exception as e:
            self.logger.error(f"Erro ao inserir clusters na tabela: {e}")
            self.session.rollback()

    def indexa_e_classifica_textos_pendentes(self) -> dict:
        """
        Processa textos pendentes: indexa no Qdrant, busca similares e forma clusters.
        """
        print_with_time(f"Iniciando indexação e detecção de textos similares...")
        self.clusters = {}  # Reseta cache
        data = self._fetch_data()
        processados = []

        for row in tqdm(data, desc="Processando textos pendentes"):
            id_texto = row['id']
            texto = row['Text']

            # Gera embedding
            embedding = embeddingsBllModule.bllEmbeddings.generate_embedding(texto)

            # Insere no Qdrant
            self._insert_qdrant(id_texto, embedding)

            # Marca como indexado
            self._mark_as_indexado(id_texto)

            # Busca similares
            high_similars = self._search_qdrant(embedding, id_texto)

            # Atualiza clusters se >= 3 similares
            if len(high_similars) >= self.min_similars:
                self._update_clusters(id_texto, high_similars)

            processados.append(id_texto)

        # Insere clusters na tabela
        self._insert_clusters_to_db()

        sucessMessage = f"Processados {len(processados)} textos pendentes com Qdrant."
        print_with_time(sucessMessage)

        return {
            "status": "OK",
            "processados": sucessMessage,
            "restate": f"Restam {self._get_Textos_Pendentes()[0]['TotalTextosPendentes']} textos pendentes."
        }