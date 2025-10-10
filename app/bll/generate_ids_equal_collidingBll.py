# GenerateIdsIguaisCollindgsBLL.py
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sqlalchemy.orm import Session
from common import print_with_time, print_error, get_localconfig
from bll.idIguaisBll import IdIguaisBll as IdIguaisBllModule
from bll.embeddings_generateBll import Embeddings_GenerateBll
from bll.idCollidingBll import IdCollidingBll as IdCollidingBllModule
from bll.embeddingsBll import EmbeddingsBll
from bll.classifica_textoBll import classifica_textoBll
import dbClasses.idIguais as idIguaisModule
import dbClasses.idsColidentes as idCollidingModule
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule
import qdrant_utils

class GenerateIdsIguaisCollindgs:
    def __init__(self, session: Session, localcfg):
        """
        Initialize the class for detecting similar text embeddings using Qdrant.
        Args:
            session (Session): SQLAlchemy session for database operations.
            localcfg: The localconfig module to read configuration.
        """
        self.session = session
        self.localconfig = localcfg
        self.config = localcfg.read_config()
        self.dataset_path = Path(self.config["dataset_path"])
        self.itens_limit = 50  # Number of nearest neighbors to search
        self.id_iguais_bll = IdIguaisBllModule(session)
        self.id_colliding_bll = IdCollidingBllModule(session)
        self.generate_embeddings = Embeddings_GenerateBll(session)
        self.embeddings_handler = EmbeddingsBll()
        self.qdrant_utils = Qdrant_UtilsModule()
        self.qdrant_client = self.qdrant_utils.get_client()
        self.collection_name = self.qdrant_utils.get_collection_name("final")
        self.classifica_texto = classifica_textoBll(self.embeddings_handler, session)

    def _fetch_data(self) -> list:
        try:
            data = self.generate_embeddings._fetch_data()  # Call _fetch_data from Embeddings_GenerateBll
            return data
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar dados do banco: {e}")

    def _genetare_ids_colliding(self) -> None:
        """
        Process a dataset to find colliding items (high similarity, different classes) and save to database.
        """
        similarity_threshold_colliding = 0.94
        # Load data from database
        data = self._fetch_data()
        # Create mapping of Id to data index
        json_id_to_index = {item["Id"]: idx for idx, item in enumerate(data)}
        # Process similarities
        removed_count = 0
        keep_indices = set(range(len(data)))
        lista_ids_collidentes = []
        print_with_time(f"Processando {len(data)} registros para colisões...")
        for i, item in enumerate(tqdm(data, desc="Buscando Colidentes")):
            if i not in keep_indices:
                continue
            id_tram = item["Id"]
            # Retrieve query embedding from Qdrant
            try:
                points = self.qdrant_client.retrieve(
                    collection_name=self.collection_name,
                    ids=[id_tram],
                    with_vectors=True
                )
                if not points:
                    print_with_time(f"Aviso: Id {id_tram} não encontrado no Qdrant, pulando")
                    continue
                query_embedding = np.array(points[0].vector).reshape(1, -1)
            except Exception as e:
                print_with_time(f"Erro ao recuperar embedding para Id {id_tram}: {e}")
                continue
            # Perform similarity search using classifica_textoBll
            try:
                result = self.classifica_texto.search_similarities(
                    query_embedding=query_embedding,
                    id_a_classificar=id_tram,
                    TabelaOrigem="textos_treinamento",
                    itens_limit=self.itens_limit,
                    gravar_log=False
                )
                results = result.ListaSimilaridade or []
            except Exception as e:
                print_with_time(f"Erro ao buscar similares para Id {id_tram}: {e}")
                continue

            items_to_remove = set()
            for similar_item in results:
                sim = similar_item.Similaridade or 0
                neighbor_id = similar_item.IdEncontrado
                neighbor_cod_classe = similar_item.CodClasse
                if neighbor_id == id_tram:
                    continue
                # Mark for removal if similarity exceeds threshold and different class
                if sim >= similarity_threshold_colliding and item["CodClasse"] != neighbor_cod_classe:
                    neighbor_orig_idx = json_id_to_index.get(neighbor_id)
                    if neighbor_orig_idx is None:
                        continue
                    # Check if neighbor_id is already in lista_ids_collidentes
                    already_exists = any(
                        id_collidente.IdColidente == neighbor_id
                        for id_collidente in lista_ids_collidentes
                    )
                    if not already_exists:
                        items_to_remove.add(neighbor_orig_idx)
                        lista_ids_collidentes.append(
                            idCollidingModule.IdsColidentes(
                                Id=id_tram,
                                IdColidente=neighbor_id,
                                semelhanca=float((sim or 0) * 100),
                            )
                        )
            keep_indices -= items_to_remove
            removed_count += len(items_to_remove)
        # Clear previous records
        try:
            self.id_colliding_bll.limpa_registros()
        except Exception as e:
            raise RuntimeError(f"Erro ao limpar registros da tabela idscolidentes: {e}")
        # Insert idscolidentes into database
        try:
            itens_inseridos = self.id_colliding_bll.commit_lista(lista_ids_collidentes)
            if itens_inseridos > 0:
                print_with_time(f"Inserido no banco em IdsColidentes: {itens_inseridos}")
            else:
                print_with_time("Nenhum IdsColidentes inserido, lista vazia ou erro na inserção.")
        except Exception as e:
            print_error(f"Erro ao inserir IdsColidentes no banco: {e}")
            raise
        print_with_time(f"Processamento de colisões concluído")
        print_with_time(f"Registros removidos: {removed_count}")

    def _generate_ids_equal(self) -> None:
        """
        Process a dataset to find similar items (high similarity, same class) and save to database.
        """
        similarity_threshold_equal = 0.985
        # Load data from database
        data = self._fetch_data()
        # Create mapping of Id to data index
        json_id_to_index = {item["Id"]: idx for idx, item in enumerate(data)}
        # Process similarities
        removed_count = 0
        keep_indices = set(range(len(data)))
        lista_ids_iguais = []
        print_with_time(f"Processando {len(data)} registros para itens iguais...")
        for i, item in enumerate(tqdm(data, desc="Buscando Similares")):
            if i not in keep_indices:
                continue
            
            id_tram = item["Id"]       
            reg = qdrant_utils.get_id(id_tram, self.collection_name)
            if reg is None:                
                continue
            else:
                query_embedding = reg["Embedding"]
            
            # Perform similarity search using classifica_textoBll
            try:
                result = self.classifica_texto.search_similarities(
                    query_embedding=query_embedding,
                    id_a_classificar=id_tram,
                    TabelaOrigem="textos_treinamento",
                    itens_limit=self.itens_limit,
                    gravar_log=False
                )
                results = result.ListaSimilaridade or []
            except Exception as e:
                print_with_time(f"Erro ao buscar similares para Id {id_tram}: {e}")
                continue

            items_to_remove = set()
            for similar_item in results:
                sim = similar_item.Similaridade or 0
                neighbor_id = similar_item.IdEncontrado
                neighbor_cod_classe = similar_item.CodClasse
                if neighbor_id == id_tram:
                    continue

                # Mark for removal if similarity exceeds threshold and same class
                if sim >= similarity_threshold_equal and item["CodClasse"] == neighbor_cod_classe:
                    neighbor_orig_idx = json_id_to_index.get(neighbor_id)
                    if neighbor_orig_idx is None:
                        continue
                    items_to_remove.add(neighbor_orig_idx)
                    lista_ids_iguais.append(
                        idIguaisModule.IdsIguais(id=id_tram, idIgual=neighbor_id)
                    )
            keep_indices -= items_to_remove
            removed_count += len(items_to_remove)

        # Clear previous records
        try:
            self.id_iguais_bll.limpa_registros()
        except Exception as e:
            raise RuntimeError(f"Erro ao limpar registros da tabela idsiguais: {e}")
        # Insert duplicates into database
        try:
            itens_inseridos = self.id_iguais_bll.commit_lista(lista_ids_iguais)
            if itens_inseridos > 0:
                print_with_time(f"IdsIguais inseridos no banco em IdsIguais: {itens_inseridos}")
            else:
                print_with_time("Nenhum IdsIguais inserido, lista vazia ou erro na inserção.")
        except Exception as e:
            print_error(f"Erro ao inserir IdsIguais no banco: {e}")
            raise
        print_with_time(f"Processamento de Ids Iguais concluído")
        print_with_time(f"Registros removidos: {removed_count}")

    def generate_ids_iguais_start(self):
        """
        Start processing the dataset for equal IDs.
        """
        print_with_time(f"Iniciando processamento de search_ids_iguais...")
        self._generate_ids_equal()
        print_with_time("Processamento completo! Execute generate_ids_iguais para gerar ids iguais.")

    def generate_ids_colliding_start(self):
        """
        Start processing the dataset for colliding IDs.
        """
        print_with_time(f"Iniciando processamento de generate_ids_colliding...")
        self._genetare_ids_colliding()
        print_with_time("Processamento completo! Execute generate_ids_colliding_start para remover ids colidentes.")