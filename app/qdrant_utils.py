# qdrant_utils.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, Filter, FieldCondition, MatchValue
from sympy import true
from common import print_with_time, print_error, get_localconfig
import re
import requests
from importlib.metadata import version as pkg_version


class Qdrant_Utils:
    def __init__(self):
        from main import localconfig as localcfg
        # Inicializa Qdrant Client
        self._Qdrant_url = localcfg.get("vectordatabasehost")
        self._qdrant_client = None
        self.CollectionSize = localcfg.get("max_length")  # Dimensão dos embeddings
        self._connect_qDrant()

    def _connect_qDrant(self) -> bool:
        try:
            self._qdrant_client = QdrantClient(url=self._Qdrant_url, timeout=60)
            print_with_time(f"QdrantClient inicializado com URL: {self._Qdrant_url}")

            # checar compatibilidade client x server
            self._check_client_server_compatibility()

            return True
        except Exception as e:
            raise RuntimeError(f"[ERRO] Falha ao conectar no qDrant: {e}")
        
    #obtem o cliente do qdrant  
    def get_client(self):
        """Cria uma nova sessão (útil para contextos paralelos)."""
        if self._qdrant_client is None:
            raise RuntimeError("Não conectado no vectordatabase não inicializado.")
        return self._qdrant_client

    def dispose(self):
        """Fecha a conexão com o Qdrant."""
        if self._qdrant_client is not None:
            self._qdrant_client.close()  # Use close() instead of dispose()
            self._qdrant_client = None

    def create_collection(self, pCollection_name: str):
        try:
            # Verifica se a coleção existe senão cria
            collections = self._qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            if pCollection_name not in collection_names:
                self._qdrant_client.create_collection(
                    collection_name=pCollection_name,
                    vectors_config=models.VectorParams(size=self.CollectionSize, distance=Distance.DOT)
                )
                print_with_time(f"Criada collection Qdrant: {pCollection_name}")

        except Exception as e:
            raise RuntimeError(f"Erro criando create_collection em Qdrant_Utils: {e}")


    # -----------------------------
    # Utilitários de versão
    # -----------------------------
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
        base = self._Qdrant_url.rstrip("/")

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
                f"[AVISO] Não foi possível detectar a versão do servidor Qdrant em {self._Qdrant_url}. "
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

