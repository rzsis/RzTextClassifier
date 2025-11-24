# qdrant_utils.py
from ast import Dict
from socket import timeout
from typing import Any, List, Optional
import venv
from xmlrpc.client import boolean
from aiohttp import Payload
import numpy as np
from pymysql import connect
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, Filter, FieldCondition, MatchValue,MatchExcept, MatchAny
from sympy import false, true
from common import print_with_time, print_error, get_localconfig
import re
import requests
from importlib.metadata import version as pkg_version
from qdrant_client.http.exceptions import UnexpectedResponse
import time

class Qdrant_Utils:
    def __init__(self):
        from main import localconfig as localcfg
        # Inicializa Qdrant Client
        self._qdrant_url = localcfg.get("vectordatabasehost")
        self._qdrant_client = None
        self.collectionSize = localcfg.get("max_length")  # Dimensão dos embeddings
        self._connect_qDrant()
        self._old_exclusion_list = []  # Cache para lista de exclusão antiga
        self._oldFilter = None

    def _connect_qDrant(self) -> bool:
        try:
            if self.connected():
                return True
            
            self._qdrant_client = QdrantClient(url=self._qdrant_url, timeout=60)
            print_with_time(f"QdrantClient inicializado com URL: {self._qdrant_url}")
          
            self._check_client_server_compatibility()  # checar compatibilidade client x server

            return True
        except Exception as e:
            raise RuntimeError(f"[ERRO] Falha ao conectar no qDrant: {e}")
        
    #verifica se o cliente esta conectado
    def connected(self) -> bool:        
        try:
            health = self._qdrant_client.get_health()
            # Alguns servidores retornam {'status': 'ok'} ou algo similar
            if isinstance(health, dict) and health.get("status") == "ok":
                return True
            return False
        except (ConnectionError, UnexpectedResponse, Exception):
            return False
            
    #obtem o cliente do qdrant  
    def get_client(self):
        """Cria uma nova sessão (útil para contextos paralelos)."""
        if self._qdrant_client is None:
            raise RuntimeError("Não conectado no vectordatabase não inicializado.")
        return self._qdrant_client

    #facilita a obtenção do nome da coleção
    def get_collection_name(self, collection_type: str) -> str:
        from main import localconfig as localcfg
        codcli = localcfg.get("codcli")
        if (collection_type == "final"):
            return f"v{codcli}_textos_final"
        elif (collection_type == "train"):
            return f"v{codcli}_textos_classificar"
        else:
            raise RuntimeError(f"get_collection_name só suporta 'final' ou 'train', recebeu: {collection_type}")
        
    ##Fecha a conexão com o Qdrant.
    def dispose(self):
        if self._qdrant_client is not None:
            self._qdrant_client.close()  # Use close() instead of dispose()
            self._qdrant_client = None

    #para evitar tentar ganhar perfomance e não montar a lista de exclusão toda hora
    def _get_exclusion_list(self, exclusion_list: List[int]) -> Filter:
        if (exclusion_list) and (exclusion_list != self._old_exclusion_list):
            self._old_exclusion_list  = exclusion_list            
            self._oldFilter = Filter(
                must_not=[
                    FieldCondition(
                        key="id",
                        match=MatchAny(any=exclusion_list)
                    )
                ]
            )
            return self._oldFilter
        else:   
            return self._oldFilter       # type: ignore
            
    
    #Cria a coleção no Qdrant se não existir
    def create_collection(self, pCollection_name: str):
        try:
            # Verifica se a coleção existe senão cria
            collections = self._qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            if pCollection_name not in collection_names:
                self._qdrant_client.create_collection(
                    collection_name=pCollection_name,
                    vectors_config=models.VectorParams(size=self.collectionSize,
                                                       distance=Distance.COSINE  
                                                    )
                )
                print_with_time(f"Criada collection Qdrant: {pCollection_name}")

        except Exception as e:
            raise RuntimeError(f"Erro criando create_collection em Qdrant_Utils: {e}")

    #Força o Qdrant a reindexar a coleção imediatamente.
    #Ele compacta segmentos e recria índices HNSW se necessário.
    def force_reindex(self, collection_name: str) -> bool:   
        try:
            # 1. Verifica status atual
            timeout = 300
            info = self._qdrant_client.get_collection(collection_name)
            total_points = info.points_count or 0
            indexed_points = info.indexed_vectors_count or 0
            
            if indexed_points >= total_points:            
                return True

            print_with_time(f"Forçando reindexação: {indexed_points}/{total_points} vetores indexados.")

            # 2. Define threshold = 0 (acumula tudo sem indexar imediatamente)
            self._qdrant_client.update_collection(
                collection_name=collection_name,
                optimizer_config={
                    "indexing_threshold": 0
                }
            )

            # 3. Volta o threshold para um valor baixo → força criação do índice
            self._qdrant_client.update_collection(
                collection_name=collection_name,
                optimizer_config={
                    "indexing_threshold": 10000  # valor pequeno o suficiente para disparar imediatamente
                }
            )

            start_time = time.time()
            
            while True:
                if time.time() - start_time > timeout:
                    print_with_time("\nTimeout atingido ao aguardar indexação.")
                    return False

                current = self._qdrant_client.get_collection(collection_name)
                indexed = current.indexed_vectors_count or 0
                total = current.points_count or 0

                if indexed >= total and current.status == "green":
                    print_with_time(f"\nIndexação concluída! {indexed}/{total} vetores indexados.")
                    return True
              
                time.sleep(2)

        except Exception as e:
            print_with_time(f"\nErro ao forçar reindexação: {e}")
            return False


    # Busca embeddings similares no qdrant
    #exclusion_list é uma lista de Ids que devem ser excluidos da busca de similaridade pois não faz sentido eu procurar os proprios ids como similares deles mesmos
    def search_embedding(self, 
                         embedding: np.ndarray,
                         collection_name: str,
                         itens_limit: int,
                         similarity_threshold: float,
                         exclusion_list: List[int]) -> list[dict]:
        try: 
            high_similars = []         
            embedding = np.array(embedding, dtype=float)  

            # Criar filtro de exclusão de IDs
            search_filter = self._get_exclusion_list(exclusion_list)            

            search_results = self._qdrant_client.search(
                collection_name=collection_name,
                query_vector=embedding.flatten().tolist(),
                limit=itens_limit,
                score_threshold=similarity_threshold,
                query_filter=search_filter 
            )


            high_similars = [
                {
                    "IdEncontrado": int(res.id),  
                    "Similaridade": res.score, # type: ignore
                    "Classe": (res.payload or {}).get("Classe"),                
                    "CodClasse": (res.payload or {}).get("CodClasse")                 
                }
                for res in search_results                
            ]
            

            return high_similars
        
        except Exception as e:
            print_with_time(f"Erro ao buscar similares no Qdrant {e}")
            return []
        
    def get_id(self, id: int, collection_name: str) -> Optional[dict[str, Any]]:
        try:
            records = self._qdrant_client.retrieve(
                collection_name=collection_name,
                ids=[id],
                with_vectors=True,
                with_payload=["Classe", "CodClasse"]
            )
            if not records:
                print_with_time(f"Aviso: Id {id} não encontrado no Qdrant, pulando")
                return None

            rec = records[0]
            vec = rec.vector
            if isinstance(vec, dict):
                vec = vec.get(self.vector_name)  # por ex. "text"            
                
            if vec is None:
                print_with_time(f"Aviso: Id {id} não possui vetor no Qdrant, pulando")
                return None

            try:
                embedding = [float(x) for x in vec]

                if len(embedding) != self.collectionSize:
                    print_with_time(f"Aviso: Vetor do Id {id} tem dimensão {len(embedding)}, esperado {self.collectionSize}")
                    return None
            except Exception:
                print_with_time(f"Aviso: Vetor do Id {id} não é numérico, pulando")
                return None

            payload = rec.payload or {}
            return {
                "IdEncontrado": int(rec.id),
                "Classe": payload.get("Classe"),
                "CodClasse": payload.get("CodClasse"),
                "Embedding": np.array(embedding, dtype=np.float32)
            }
        except Exception as e:
            print_with_time(f"Erro ao recuperar embedding para Id {id}: {e}")
            return None
    
    def delete_id(self,
                  collection_name:str,
                  id:int):
        try:
            self._qdrant_client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=[id])
            )
        except Exception as e:
            print_with_time(f"Erro ao apagar Id {id}: {e}")

    #Insere ou atualiza um ID no qdrant
    def upinsert_id(self,collection_name:str, id:int, embeddings: np.ndarray, codclasse:int, classe:str ) -> bool:
        try:
            #só deve obrigar codclasse e classe na coleção final
            final_collection = self.get_collection_name("final")

            if (collection_name == final_collection)  and ((codclasse is None) or (codclasse == 0)):
                raise RuntimeError(f"CodClasse não pode ser None ou 0 para o ID {id} na coleção {collection_name}")
            
            if (collection_name == final_collection)  and ((classe is None) or (classe.strip() == "")): 
                raise RuntimeError(f"Classe não pode ser None para o ID {id} na coleção {collection_name}")
            
            #embeddings não pode ser None ou vazio
            if (embeddings is None) or (len(embeddings) == 0):
                raise RuntimeError(f"Embeddings não pode ser None ou vazio para o ID {id} na coleção {collection_name}")

            if len(embeddings) != self.collectionSize:
                raise RuntimeError(f"Embeddings para o ID {id} na coleção {collection_name} tem tamanho {len(embeddings)}, esperado {self.collectionSize}")

            payload={        
                    "Id": id,
                    "Classe": classe,
                    "CodClasse": codclasse                        
                    }

            self._qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=id,
                        vector=embeddings,
                        payload=payload
                    )
                ]
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Erro ao inserir ID {id} na coleção {collection_name}: {e}")

    # Utilitários de versão
    def _parse_semver(self, v: str) -> tuple[int, int, int]:
        """
        Converte '1.15.1' em (1, 15, 1). Ignora sufixos como '-rc', '-beta' etc.
        """
        if not v:
            return (0, 0, 0)
        m = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", v)
        if not m:
            return (0, 0, 0)
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    def _get_server_version(self) -> str:
        """
        Obtém a versão do servidor Qdrant priorizando /telemetry (onde seu servidor retorna),
        com fallbacks em /version, headers de /collections e /.
        Retorna "" se não conseguir detectar.
        """
        base = self._qdrant_url.rstrip("/")

        # 1) /telemetry  → espera-se "result.app.version"
        try:
            r = requests.get(f"{base}/telemetry", timeout=5)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and isinstance(data.get("result"), dict):
                app = data["result"].get("app")
                if isinstance(app, dict) and isinstance(app.get("version"), str):
                    return app["version"].strip()

            # fallback: varredura por qualquer chave "version" com semver
            def _scan_version(node):
                if isinstance(node, dict):
                    for k, v in node.items():
                        if k == "version" and isinstance(v, str) and re.match(r"^\d+\.\d+\.\d+", v):
                            return v.strip()
                        if isinstance(v, (dict, list)):
                            found = _scan_version(v)
                            if found:
                                return found
                elif isinstance(node, list):
                    for v in node:
                        found = _scan_version(v)
                        if found:
                            return found
                return ""

            any_ver = _scan_version(data)
            if any_ver:
                return any_ver
        except Exception:
            pass

        # 2) /version (algumas versões expõem)
        try:
            r = requests.get(f"{base}/version", timeout=5)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                if isinstance(data.get("version"), str):
                    return data["version"].strip()
                if isinstance(data.get("result"), dict):
                    v = data["result"].get("version")
                    if isinstance(v, str):
                        return v.strip()
        except Exception:
            pass

        # helper para extrair header
        def _get_ver_from_headers(resp):
            hdr = resp.headers.get("x-qdrant-version") or resp.headers.get("X-Qdrant-Version")
            return hdr.strip() if isinstance(hdr, str) else ""

        # 3) headers em /collections
        try:
            r = requests.get(f"{base}/collections", timeout=5)
            r.raise_for_status()
            v = _get_ver_from_headers(r)
            if v:
                return v
        except Exception:
            pass

        # 4) headers na raiz /
        try:
            r = requests.get(base + "/", timeout=5)
            r.raise_for_status()
            v = _get_ver_from_headers(r)
            if v:
                return v
        except Exception:
            pass

        # Não conseguiu detectar
        return ""

    def _check_client_server_compatibility(self):
        """
        Verifica se a versão do client é compatível com a do servidor.
        Regra do Qdrant: major iguais e |minor_client - minor_server| <= 1.
        Em caso de incompatibilidade, levanta RuntimeError com a mensagem padrão.
        """
        client_ver = pkg_version("qdrant-client")
        server_ver = self._get_server_version()

        # Se não conseguirmos obter a versão do servidor, não bloqueia — apenas informa.
        if not server_ver:
            print_with_time(
                f"[AVISO] Não foi possível detectar a versão do servidor Qdrant em {self._qdrant_url}. "
                f"Versão do client: {client_ver}"
            )
            return

        cM, cm, _ = self._parse_semver(client_ver)
        sM, sm, _ = self._parse_semver(server_ver)

        incompatible = (cM != sM) or (abs(cm - sm) > 1)
        if incompatible:
            # Mensagem seguindo o padrão mostrado no warning oficial
            msg = (
                f"Qdrant client version {client_ver} is incompatible with server version {server_ver}. "
                f"Major versions should match and minor version difference must not exceed 1."
            )
            # Levanta RuntimeError, incluindo a versão do servidor
            raise RuntimeError(msg)

        # Caso compatível, apenas log informativo
        print_with_time(f"Compatibilidade qdrant com client ok: client {client_ver} ~ server {server_ver}")

