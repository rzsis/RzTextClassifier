#embeddings_generate.py
import os
from pathlib import Path
from typing_extensions import runtime
import numpy as np
from requests import Session
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from common import  print_with_time, print_error
from collections import Counter
from bll.idsDuplicadosBll import IdsDuplicados
from sqlalchemy import text
import gpu_utils as gpu_utilsModule
import localconfig

class Embeddings_GenerateBll:
    def __init__(self, split:str, session: Session, localcfg:localconfig):
        self.session        = session
        self.localconfig    = localcfg
        self.config         = localcfg.read_config()  # Read config from the provided localconfig module
        self.embeddingsTrain   = localcfg.getEmbeddingsTrain()                
        self.model_path     = Path(self.config["model_path"])
        self.max_length     = self.config["max_length"]
        self.batch_size     = int(self.config["batch_size"])
        self.textual_fields = {'Text'}  # Fields for embeddings
        self.metadata_fields = {'Classe', 'Id', 'CodClasse', 'QtdItens'}  # Include QtdItens
        self.tokenizer      = None
        self.model          = None  
        self.split          = split  # 'train' or 'final'      
        self.ids_duplicados = IdsDuplicados(session)
        self.gpu_utils      = gpu_utilsModule.GpuUtils()

        # Validate model directory
        if not os.path.isdir(self.model_path):
            print_with_time(f"Error: Model directory not found: {self.model_path}")
            raise RuntimeError(f"Model directory not found: {self.model_path}")

        # Load tokenizer and model
        self._load_model()

    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            raise RuntimeError("verificar pois isso deveria de carregar o modelo e tokenizer")
        
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), use_fast=True)
            self.model = AutoModel.from_pretrained(str(self.model_path)).to("cuda" if torch.cuda.is_available() else "cpu")
            self.model.eval()
            if torch.cuda.is_available():
                self.model.half()
            if not self.tokenizer.is_fast:
                print_with_time("Warning: Using slow tokenizer, performance may be reduced")
        except Exception as e:
            print_with_time(f"Error loading tokenizer or model: {e}")
            raise RuntimeError(f"Error loading tokenizer or model: {e}")

    def _validate_example(self, exemplo):
        """
        Validate an example based on defined textual and metadata fields.

        Args:
            exemplo (dict): Example from the dataset.

        Returns:
            bool: True if the example is valid, False otherwise.
        """
        if not isinstance(exemplo, dict):
            print_with_time(f"Ignoring invalid example (not a dictionary): {exemplo}")
            return False

        all_fields = self.textual_fields | self.metadata_fields
        for field in all_fields:
            if field not in exemplo:
                print_with_time(f"Ignoring example missing field '{field}': {exemplo}")
                return False

            value = exemplo[field]
            if field in self.textual_fields:
                if not isinstance(value, str) or not value.strip():
                    print_with_time(f"Ignoring example with invalid/empty textual field '{field}': {exemplo}")
                    return False
            elif value is None:
                print_with_time(f"Ignoring example with null metadata field '{field}': {exemplo}")
                return False

        return True

    def _get_embedding_batch(self, texts):
        """
        Generate embeddings for a batch of texts.

        Args:
            texts (list): List of text strings.

        Returns:
            np.ndarray: Embeddings for the batch.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        ).to(self.model.device) # pyright: ignore[reportOptionalCall]

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast(device_type='cuda'):
                    outputs = self.model(**inputs) # pyright: ignore[reportOptionalCall]
            else:
                outputs = self.model(**inputs) # pyright: ignore[reportOptionalCall]

            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def _fecth_data(self) -> list:
        # Define SQL query (deterministic with MIN(t.id))
        query = """
            SELECT MIN(t.id) AS Id,
                   c.CodClasse,
                   c.Classe,
                   t.TxtTreinamento AS Text,
                   COUNT(t.id) AS QtdItens
            FROM textos_treinamento t
            INNER JOIN classes c ON c.CodClasse = t.CodClasse
            WHERE LENGTH(TRIM(t.TxtTreinamento)) > 0
            AND t.CodClasse IS NOT NULL 
            AND not t.id in (Select id from idsduplicados)                        
            AND not t.id in (Select id from idsiguais)                                    
            GROUP BY t.TxtTreinamento, t.CodClasse, c.Classe
            Order by COUNT(t.id),t.id DESC  /*ordena para inserir duplicatas primeiro em idduplicados para para poder ignorar depois*/           
        """

        # Fetch data from database
        try:
        
            result = self.session.execute(text(query)).mappings().all()
            dados = [dict(row) for row in result]
            if not dados:            
                raise RuntimeError("Não ha dados para gerar o modelo de embeddings.")                    
                        
            return dados        
        except Exception as e:            
            raise RuntimeError(f"Erro executando consulta no banco de dados: {e}")
        
    #Transforma os dados em embeddings
    def _process_embeddings(self, dados, split) -> tuple:
        print_with_time(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")        
        # Prepare structures
        field_embeddings = {field: [] for field in self.textual_fields}
        field_metadata = {field: [] for field in self.metadata_fields}
        skipped = 0
        invalid_reasons = Counter()

        print_with_time(f"Processando ({len(dados)} registros) e lotes de {self.batch_size} para o banco {split}...")

        # Process in batches
        for i in tqdm(range(0, len(dados), self.batch_size), desc="Generating embeddings in batches"):
            batch_dados = dados[i:i + self.batch_size]
            batch_valid = []

            # Filter valid examples
            for exemplo in batch_dados:
                if self._validate_example(exemplo):
                    batch_valid.append(exemplo)
                else:
                    skipped += 1
                    invalid_reasons['invalid_example'] += 1

            if invalid_reasons:
                print_with_time(f"Batch {i//self.batch_size}: Skipped {sum(invalid_reasons.values())} examples: {invalid_reasons}")

            if not batch_valid:
                continue

            try:
                # Collect texts for textual fields
                batch_texts = {field: [ex[field] for ex in batch_valid if field in ex] for field in self.textual_fields}

                # Generate embeddings
                temp_embeddings = {}
                for field in self.textual_fields:
                    texts = batch_texts[field]
                    if texts:
                        embeddings = self._get_embedding_batch(texts)
                        temp_embeddings[field] = embeddings

                # Append embeddings
                for field in self.textual_fields:
                    if field in temp_embeddings:
                        field_embeddings[field].extend(temp_embeddings[field])

                # Append metadata and check for duplicates
                for field in self.metadata_fields:
                    batch_values = [ex.get(field) for ex in batch_valid]
                    field_metadata[field].extend(batch_values)

                # Check QtdItens for duplicates
                for exemplo in batch_valid:
                    if exemplo['QtdItens'] > 1:
                        self.ids_duplicados.insert_duplicate_ids(
                            id=exemplo['Id'],
                            texto=exemplo['Text'],
                            cod_classe=exemplo['CodClasse']
                        )

                # Clear cache every X batches
                if i % 10 == 0:
                   self.gpu_utils.clear_gpu_cache()

            except torch.cuda.CudaError as e:
                print_error(f"CUDA error in batch {i//self.batch_size}: {e}")
                self.gpu_utils.clear_gpu_cache()
                skipped += len(batch_valid)
                continue
            except Exception as e:
                print_error(f"Error in batch {i//self.batch_size}: {e}")
                skipped += len(batch_valid)
                continue

        # Report
        print_with_time(f"Split : {len(dados) - skipped} examples processed, {skipped} examples skipped.")

        # Validate textual fields
        if not self.textual_fields:            
            raise RuntimeError("No textual fields defined")       
        
        # Validate
        if not field_embeddings:
            raise RuntimeError("Error: No textual fields processed for embeddings")

        lengths = {field: len(embs) for field, embs in field_embeddings.items()}
        metadata_lengths = {field: len(values) for field, values in field_metadata.items()}

        for field, length in lengths.items():
            if length == 0:                
                raise RuntimeError(f"Textual field '{field}' has 0 entities")

        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:            
            raise RuntimeError(f"Textual fields have different entity counts: {lengths}")

        for field, length in metadata_lengths.items():
            if length == 0 or length != list(lengths.values())[0]:                
                raise RuntimeError(f"Metadata '{field}' has incorrect length: {length}")

        # Convert to NumPy
        field_embeddings['Text'] = np.vstack(field_embeddings["Text"])
        for field in field_metadata:
            field_metadata[field] = np.array(field_metadata[field], dtype=object)

        return field_embeddings, field_metadata 

    def _create_embeddings_backup(self, split):
        backup_dir = os.path.join(self.embeddingsTrain, "backup")
        os.makedirs(backup_dir, exist_ok=True)

        embeddings_file = os.path.join(self.embeddingsTrain, f"{split}_text.npy")
        metadata_file = os.path.join(self.embeddingsTrain, f"{split}_metadata.npz")

        if os.path.exists(embeddings_file):
            backup_embeddings_file = os.path.join(backup_dir, f"{split}_text_backup.npy")
            os.rename(embeddings_file, backup_embeddings_file)
            print_with_time(f"Backup created for embeddings: {backup_embeddings_file}")

        if os.path.exists(metadata_file):
            backup_metadata_file = os.path.join(backup_dir, f"{split}_metadata_backup.npz")
            os.rename(metadata_file, backup_metadata_file)
            print_with_time(f"Backup created for metadata: {backup_metadata_file}")


    # Save embeddings and metadata to disk
    def _save_embeddings(self,split,field_embeddings,field_metadata):
        self._create_embeddings_backup( split)  # Create backup if files exist
        if (split == 'train'):
            dest_path = self.embeddingsTrain    
        else:
            dest_path = self.embeddingsfinal

        os.makedirs(dest_path, exist_ok=True)

        embeddings_file = os.path.join(dest_path, f"{split}_text.npy")
        try:
            np.save(embeddings_file, field_embeddings['Text'])
            print_with_time(f"Embeddings for text saved to: {embeddings_file}")
        except Exception as e:
            print_with_time(f"Error saving embeddings for text: {e}")
            raise RuntimeError(f"Error saving embeddings: {e}")

        # Save metadata
        metadata_file = os.path.join(self.embeddingsTrain, f"{split}_metadata.npz")
        try:
            np.savez(metadata_file, **field_metadata)
            print_with_time(f"Metadata (including {self.metadata_fields}) saved to: {metadata_file}")
        except Exception as e:
            print_with_time(f"Error saving metadata: {e}")
            raise RuntimeError(f"Error saving metadata: {e}")

        print_with_time(f"✅ Split '{split}' processado com sucesso")

    

    def start(self):
        """
        Start the embedding generation process for the specified split.

        Args:
            split (str): Data split to process (default: 'train').
        """

        dados = self._fecth_data() #busca os dados no banco de dados
     
        field_embeddings,field_metadata = self._process_embeddings(dados,self.split)# processa os dados e transforma em embeddings
            
        self._save_embeddings(self.split,field_embeddings,field_metadata)# salva em disco os embeddings e metadados