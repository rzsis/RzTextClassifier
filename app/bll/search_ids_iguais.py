#search_ids_iguais.py
import os
from pathlib import Path
import numpy as np
import faiss
import json
from tqdm import tqdm
from sqlalchemy.orm import Session
from common import print_with_time, print_error, get_localconfig
from bll.idIguaisBll import IdIguaisBll
from bll.embeddings_generate import GenerateEmbeddings
from bll.embeddings import Embenddings
import dbClasses.idIguais as idIguais

class SearchIdIguais:
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
        self.embeddings_dir = Path(localcfg.getEmbendingPath())
        self.output_dir = self.dataset_path
        self.field = "text"
        self.k = 50  # Number of nearest neighbors to search
        self.similarity_threshold = 0.99  # Similarity threshold for marking duplicates
        self.id_iguais_bll = IdIguaisBll(session)
        self.generate_embeddings = GenerateEmbeddings(session, localcfg)
        self.embeddings_handler = Embenddings(localcfg)

    def _fetch_data(self) -> list:        
        try:
            data = self.generate_embeddings._fecth_data()  # Call _fetch_data from GenerateEmbeddings
            return data
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar dados do banco: {e}")            

    def _load_embeddings_and_metadata(self) -> tuple:
        try:
            embeddings, metadata = self.embeddings_handler.load_model_and_embendings("train")  # Load training embeddings
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

    def _process_dataset(self) -> None:
        """
        Process a dataset to find similar items and save results.
        Args:
            dataset_name (str): Name of the dataset (e.g., 'train_final').
        """
        # Clear previous records
        try:
            self.id_iguais_bll.limpa_registros() 
        except Exception as e:            
            raise RuntimeError(f"Erro ao limpar registros da tabela idsiguais: {e}")

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
            index.add(embeddings)
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
                distances, indices = index.search(query_embedding, self.k)
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
                if sim >= self.similarity_threshold and item['CodClasse'] == neighbor_cod_classe:
                    neighbor_orig_idx = json_id_to_index.get(neighbor_id)
                    if neighbor_orig_idx is None:
                        continue
                    items_to_remove.add(neighbor_orig_idx)
                    lista_ids_iguais.append(idIguais.IdsIguais(id=id_tram, idIgual=neighbor_id))

                # Log collisions with similarity > 0.85
                if sim > 0.85:
                    collision_info.append({
                        "Id": neighbor_id,
                        "semelhanca": float(sim)
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

        # Create cleaned dataset
        cleaned_data = [data[i] for i in sorted(keep_indices)]

        # Save JSON files
        collision_file = self.output_dir / f"log_com_colisao_mesma_classe.json"
        self._save_json(output_data, collision_file)

        clean_file = self.output_dir / f"log_sem_colisao_mesma_classe.json"
        self._save_json(cleaned_data, clean_file)

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

        print_with_time(f"Processamento de busca concluida")
        print_with_time(f"Registros removidos: {removed_count}")
        print_with_time(f"Registros mantidos: {len(cleaned_data)}")

    def start(self):
        """
        Start processing the dataset.
        """
        print_with_time(f"Iniciando processamento de search_ids_iguais...")
        self._process_dataset()
        print_with_time("Processamento completo! Execute clean_similarity_diferent_class para remover colisões de classes diferentes.")