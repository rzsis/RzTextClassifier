# qdrant_utils.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, Filter, FieldCondition, MatchValue
from sympy import true
from common import print_with_time, print_error, get_localconfig

class Qdrant_Utils:
    def __init__(self):
        from main import localconfig as localcfg   
                  # Inicializa Qdrant Client
        self._Qdrant_url = localcfg.get("vectordatabasehost")
        self._qdrant_client = None
        self._connect_qDrant()

    def _connect_qDrant(self) -> bool:
        try:
            self._qdrant_client = QdrantClient(url=self._Qdrant_url, timeout=60)            
            print_with_time(f"QdrantClient inicializado com URL: {self._Qdrant_url}")
            return True            
        except Exception as e:
            raise RuntimeError(f"[ERRO] Falha ao conectar no qDrant: {e}")


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

    def create_collection(self, collection_name):
            try:
                # Verifica se a coleção existe senão cria
                collections = self._qdrant_client.get_collections()
                collection_names = [collection.name for collection in collections.collections]
                if collection_name not in collection_names:
                    self._qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(size=self.embedding_dim, distance=Distance.DOT)
                    )
                    print_with_time(f"Criada collection Qdrant: {collection_name}")
                    
            except Exception as e:
                raise RuntimeError(f"Erro criando create_collection em Qdrant_Utils: {e}")