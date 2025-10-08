#GenerateIdsIguaisCollindgsBLL.py
import os
from pathlib import Path
import numpy as np
import faiss
import json
from sympy import Id
from tqdm import tqdm
from sqlalchemy.orm import Session
from common import print_with_time, print_error, get_localconfig
from bll.idIguaisBll import IdIguaisBll as IdIguaisBllModule
from bll.embeddings_generateBll import Embeddings_GenerateBll
from bll.idCollidingBll import IdCollidingBll as IdCollidingBllModule
from bll.embeddingsBll import EmbeddingsBll
import dbClasses.idIguais as idIguaisModule
import dbClasses.idsColidentes as idCollidingModule


class GenerateIdsIguaisCollindgs:
    def __init__(self, session: Session, localcfg):
        """
        Initialize the FoundIdIguais class for detecting similar text embeddings.
        Args:
            session (Session): SQLAlchemy session for database operations.
            localcfg: The localconfig module to read configuration.
        """
        self.session = session
        self.localconfig = localcfg
        self.config = localcfg.read_config()
        self.dataset_path = Path(self.config["dataset_path"])
        self.embeddings_dir = Path(localcfg.getEmbeddingsTrain())
        self.log_dir = "../log"
        self.field = "text"
        self.k = 50  # Number of nearest neighbors to search
        self.id_iguais_bll = IdIguaisBllModule(session)
        self.id_colliding_bll = IdCollidingBllModule(session)
        self.generate_embeddings = Embeddings_GenerateBll('train', session, localcfg)
        self.embeddings_handler = EmbeddingsBll()

    def _fetch_data(self) -> list:        
        try:
            data = self.generate_embeddings._fecth_data()  # Call _fetch_data from GenerateEmbeddings
            return data
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar dados do banco: {e}")            

    def _load_embeddings_and_metadata(self) -> tuple:
        try:
            embeddings, metadata = self.embeddings_handler.load_model_and_embendings("train",self.session)  # Load training embeddings
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar embeddings e metadados: {e}")            

        # Create id_to_index mapping
        id_to_index = {id_tram: idx for idx, id_tram in enumerate(metadata['Id'])}
        return embeddings, metadata, id_to_index

    def _save_json(self, data: list, file_path: Path):
        """
        Save data to a JSON file.
        Args:
            data (list): Data to save.
            file_path (Path): Path to save the JSON file.
        """
        file_path = Path(file_path)
        print_with_time(f"Salvando JSON em: {file_path}")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print_error(f"Erro ao salvar JSON: {e}")
            raise RuntimeError(f"Error saving JSON: {e}")

    def _genetare_ids_colliding(self) -> None:
        similarity_threshold_colliding = 0.94
        """
        Process a dataset to find colliding items and save results.
        Args:
            dataset_name (str): Name of the dataset (e.g., 'train_final').
        """
        
        # Load data from database
        data = self._fetch_data()

        # Load embeddings and metadata
        embeddings, metadata, id_to_index = self._load_embeddings_and_metadata()

        # Create mapping of Id to data index
        json_id_to_index = {item["Id"]: idx for idx, item in enumerate(data)}

        # Check for missing IDs
        missing_ids = [item["Id"] for item in data if item["Id"] not in id_to_index]
        if missing_ids:
            print_with_time(f"Aviso: {len(missing_ids)} Ids do banco não encontrados nos embeddings: {missing_ids[:5]}...")

        # Create Faiss index
        try:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings) # pyright: ignore[reportCallIssue]
        except Exception as e:            
            raise RuntimeError(f"Erro ao criar ou adicionar ao índice Faiss: {e}")

        # Process similarities
        output_data = []
        removed_count = 0
        keep_indices = set(range(len(data)))
        lista_ids_collidentes = []

        print_with_time(f"Processando {len(data)} registros...")
        for i, item in enumerate(tqdm(data, desc="Buscando Colidentes")):
            if i not in keep_indices:
                continue

            id_tram = item["Id"]
            if id_tram not in id_to_index:
                print_with_time(f"Aviso: Id {id_tram} não encontrado nos embeddings, pulando")
                continue

            # Perform k-NN search
            try:
                query_embedding = embeddings[id_to_index[id_tram]:id_to_index[id_tram]+1]
                distances, indices = index.search(query_embedding, self.k) # pyright: ignore[reportCallIssue]
            except Exception as e:
                print_with_time(f"Erro ao buscar k-NN para Id {id_tram}: {e}")
                continue

            similarities = distances[0]
            collision_info = []
            items_to_remove = set()

            for j, (sim, neighbor_idx) in enumerate(zip(similarities, indices[0])):
                if neighbor_idx == id_to_index[id_tram]:
                    continue
                neighbor_id = metadata['Id'][neighbor_idx]
                neighbor_cod_classe = metadata['CodClasse'][neighbor_idx]

                # Mark for removal if similarity exceeds threshold and same class
                if sim >= similarity_threshold_colliding and item['CodClasse'] != neighbor_cod_classe:
                    neighbor_orig_idx = json_id_to_index.get(neighbor_id)
                    if neighbor_orig_idx is None:
                        continue
                    # Verifica se id_tram ou neighbor_id já estão na lista_ids_collidentes
                    already_exists = any(
                        # id_collidente.Id == id_tram or id_collidente.IdColidente == id_tram or
                        id_collidente.IdColidente == neighbor_id or id_collidente.IdColidente == neighbor_id
                        for id_collidente in lista_ids_collidentes
                    )
                    if not already_exists:                    
                        items_to_remove.add(neighbor_orig_idx)
                        lista_ids_collidentes.append(idCollidingModule.IdsColidentes(Id=id_tram, 
                                                                        IdColidente=neighbor_id,
                                                                        semelhanca=float((sim or 0)*100),
                                                                        ))

                    # Log collisions with similarity > similarity_threshold_colliding and classe different
                    if sim > similarity_threshold_colliding:
                        collision_info.append({
                            "Id": neighbor_id,
                            "Semelhanca": float(sim),
                            "CodClasse": neighbor_cod_classe,
                        })

            # Add item to output if it has collisions
            if collision_info:
                output_item = {
                    "IdPrincipal": id_tram,
                    "CodClasse": item["CodClasse"],
                    "IdsColidentes": collision_info
                }
                output_data.append(output_item)

            keep_indices -= items_to_remove
            removed_count += len(items_to_remove)

        # Save JSON files
        collision_file = os.path.join(self.log_dir,f"log_com_colisao_classe_diferente.json")
        self._save_json(output_data, collision_file)

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

        print_with_time(f"Processamento de colidencias concluida")
        print_with_time(f"Registros removidos: {removed_count}")        


    def _generate_ids_equal(self) -> None:
        """
        Process a dataset to find similar items and save results.
        Args:
            dataset_name (str): Name of the dataset (e.g., 'train_final').
        """
    
        similarity_threshold_equal = 0.985
        # Load data from database
        data = self._fetch_data()

        # Load embeddings and metadata
        embeddings, metadata, id_to_index = self._load_embeddings_and_metadata()

        # Create mapping of Id to data index
        json_id_to_index = {item["Id"]: idx for idx, item in enumerate(data)}

        # Check for missing IDs
        missing_ids = [item["Id"] for item in data if item["Id"] not in id_to_index]
        if missing_ids:
            print_with_time(f"Aviso: {len(missing_ids)} Ids do banco não encontrados nos embeddings: {missing_ids[:5]}...")

        # Create Faiss index
        try:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings) # pyright: ignore[reportCallIssue]
        except Exception as e:            
            raise RuntimeError(f"Erro ao criar ou adicionar ao índice Faiss: {e}")

        # Process similarities
        output_data = []
        removed_count = 0
        keep_indices = set(range(len(data)))
        lista_ids_iguais = []

        print_with_time(f"Processando {len(data)} registros...")
        for i, item in enumerate(tqdm(data, desc="Buscando similares")):
            if i not in keep_indices:
                continue

            id_tram = item["Id"]
            if id_tram not in id_to_index:
                print_with_time(f"Aviso: Id {id_tram} não encontrado nos embeddings, pulando")
                continue

            # Perform k-NN search
            try:
                query_embedding = embeddings[id_to_index[id_tram]:id_to_index[id_tram]+1]
                distances, indices = index.search(query_embedding, self.k) # pyright: ignore[reportCallIssue]
            except Exception as e:
                print_with_time(f"Erro ao buscar k-NN para Id {id_tram}: {e}")
                continue

            similarities = distances[0]
            collision_info = []
            items_to_remove = set()

            for j, (sim, neighbor_idx) in enumerate(zip(similarities, indices[0])):
                if neighbor_idx == id_to_index[id_tram]:
                    continue
                neighbor_id = metadata['Id'][neighbor_idx]
                neighbor_cod_classe = metadata['CodClasse'][neighbor_idx]

                # Mark for removal if similarity exceeds threshold and same class
                if sim >= similarity_threshold_equal and item['CodClasse'] == neighbor_cod_classe:
                    neighbor_orig_idx = json_id_to_index.get(neighbor_id)
                    if neighbor_orig_idx is None:
                        continue
                    items_to_remove.add(neighbor_orig_idx)
                    lista_ids_iguais.append(idIguaisModule.IdsIguais(id=id_tram, idIgual=neighbor_id))

                # Log collisions with similarity > 0.85
                if sim > 0.85:
                    collision_info.append({
                        "Id": neighbor_id,
                        "Semelhanca": float(sim),
                        "CodClasse": neighbor_cod_classe,
                    })

            # Add item to output if it has collisions
            if collision_info:
                output_item = {
                    "IdPrincipal": id_tram,
                    "CodClasse": item["CodClasse"],
                    "IdsColidentes": collision_info
                }
                output_data.append(output_item)

            keep_indices -= items_to_remove
            removed_count += len(items_to_remove)

        
        # Save JSON files
        collision_file = os.path.join(self.log_dir,f"log_com_colisao_mesma_classe.json")
        self._save_json(output_data, collision_file)

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
                print_with_time("Nenhum IdIguais inserido, lista vazia ou erro na inserção.")
        except Exception as e:
            print_error(f"Erro ao inserir IdsIguais no banco: {e}")
            raise

        print_with_time(f"Processamento de Ids Iguais concluido")
        print_with_time(f"Registros removidos: {removed_count}")        

    def generate_ids_iguais_start(self):
        """
        Start processing the dataset.
        """
        print_with_time(f"Iniciando processamento de search_ids_iguais...")
        self._generate_ids_equal()
        print_with_time("Processamento completo! Execute generate_ids_iguais para gerar ids iguais.")

    def generate_ids_colliding_start(self):
        """
        Start processing the dataset.
        """
        print_with_time(f"Iniciando processamento de generate ids collinding...")
        self._genetare_ids_colliding()
        print_with_time("Processamento completo! Execute generate_ids_colliding_start para remover ids colidentes.")        