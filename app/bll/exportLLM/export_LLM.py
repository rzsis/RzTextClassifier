import os
from pathlib import Path
import re
import time
import json
from collections import Counter
from sqlalchemy import text
from requests import Session
from tqdm import tqdm

import localconfig
from common import print_and_log, print_error
from qdrant_utils import Qdrant_Utils
from bll.embeddingsBll import get_bllEmbeddings
from dbClasses.classes_utils import initClassesUtils, get_ClassesUtils

class Export_LLM:
    def __init__(self, session: Session):
        self.session = session
        self.localconfig = localconfig
        self.config = localconfig.read_config()
        self.embeddings_bll = get_bllEmbeddings()
        
        # Garante que classes_utils_singleton está vivo (senão qdrant_utils.search ou get_id quebram no get_nome_classe)
        initClassesUtils(self.session)
        
        self.qdrant_utils = Qdrant_Utils()
        self.collection_name = self.qdrant_utils.get_collection_name("final")

    def _fetch_data(self) -> list:
        # Pega todos os textos de treinamento disponíveis.
        # Caso tenha outras restrições semelhantes ao do DAPT (ex. QtdPalavras), aplicar aqui
        query = f"""
                SELECT t.id, t.TxtTreinamento, t.CodClasse
                FROM textos_treinamento t
                WHERE t.TxtTreinamento <> ''
                ORDER BY t.id ASC
        """
        try:
            result = self.session.execute(text(query)).mappings().all()
            return [dict(row) for row in result]
        except Exception as e:
            raise RuntimeError(f"Erro executando consulta de textos de treinamento: {e}")

    def _is_similar(self, text_id: int, embedding: list, cod_classe: int, exported_ids: set) -> bool:
        """
        Verifica se já existe um texto muito similar (>97%) da mesma classe
        que já foi incluído na exportação atual.
        """
        try:
            # Note que id=0 é para ignorar o id search. codclasse=cod_classe para forçar mesma classe
            similares = self.qdrant_utils.search_embedding_and_metaData(
                embedding=embedding,
                collection_name=self.collection_name,
                itens_limit=1000,
                similarity_threshold=97.0,  # pois a BLL de qdrant_utils divide por 100
                id=0,
                codclasse=cod_classe
            )

            for sim in similares:
                # O search já filtrou pela mesma classe e cut-off de 97% de similaridade.
                # Se o similar encontrado for diferente deste text_id E o similar já tiver sido exportado, 
                # então este text_id deve ser descartado pois é redundante na exportação.
                similar_id = sim["IdEncontrado"]
                if similar_id != text_id and similar_id in exported_ids:
                    return True
            
            return False

        except Exception as e:
            print_and_log(f"Erro ao buscar similares para o id {text_id}: {e}")
            return False

    def _build_full_dataset(self, dados: list) -> list:
        dataset_full = []
        for row in dados:
            txt = row["TxtTreinamento"].strip()
            if not txt:
                continue

            cod_classe = row["CodClasse"]
            dataset_full.append({
                "text": txt,
                "label": cod_classe,
                "label_text": get_ClassesUtils().get_nome_classe(cod_classe),
                "id": row["id"]
            })

        return dataset_full

    def _save_datasets(self, output_path: str, dataset_train: list, dataset_full: list) -> tuple:
        codcli = self.localconfig.get("codcli")
        train_file_path = os.path.join(output_path, f"{codcli}_dataset_train.json")
        full_file_path = os.path.join(output_path, f"{codcli}_dataset_full.json")

        with open(train_file_path, "w", encoding="utf-8") as f:
            json.dump(dataset_train, f, ensure_ascii=False, indent=2)

        with open(full_file_path, "w", encoding="utf-8") as f:
            json.dump(dataset_full, f, ensure_ascii=False, indent=2)

        return train_file_path, full_file_path

    def start(self):
        iniTime = time.time()
        print_and_log(f"Iniciando exportação de dados para Dataset LLM: {iniTime}")

        dados = self._fetch_data()
        qtdreg = len(dados)

        if qtdreg == 0:
            return {"status": "Completo", "message": "Não há dados para exportar."}

        print_and_log(f"Total de registros a processar: {qtdreg}")
        
        tmpErros = ""
        processados = 0
        exportados = 0
        dataset = []
        dataset_full = self._build_full_dataset(dados)
        exported_ids = set()

        for row in tqdm(dados, desc="Avaliando similaridade para Exportação LLM"):
            txt = row['TxtTreinamento'].strip()
            text_id = row["id"]
            cod_classe = row["CodClasse"]

            if not txt:
                processados += 1
                continue

            try:
                embedding = None
                
                # Resgata do Qdrant primeiro para poupar reprocessamento
                vdb_record = self.qdrant_utils.get_id(text_id, self.collection_name)
                if vdb_record and vdb_record.get("Embedding") is not None:
                    embedding = vdb_record["Embedding"]
                else:
                    # Gera o embedding para um texto único se não existir no BD vetorial
                    embeddings = self.embeddings_bll.generate_embeddings([txt], [text_id])
                    if embeddings and len(embeddings) > 0:
                        embedding = embeddings[0]
                
                if embedding is not None and len(embedding) > 0:
                    # Verifica se o texto atual colide fortemente com algum já exportado na mesma classe
                    if not self._is_similar(text_id, embedding, cod_classe, exported_ids):
                        dataset.append({
                            "text": txt,
                            "label": cod_classe,
                            "label_text": get_ClassesUtils().get_nome_classe(cod_classe),
                            "id": text_id
                        })
                        exported_ids.add(text_id)
                        exportados += 1
                
                processados += 1
            except Exception as e:
                tmpErros += f"Erro ao processar item com id {text_id}: {e}\n"
                print_and_log(f"Erro no processamento do item {text_id}: {e}")
                continue

        # Salvando em arquivo
        output_path = self.localconfig.get("dataset_path")
        os.makedirs(output_path, exist_ok=True)

        result_msg = ""
        try:
            train_file_path, full_file_path = self._save_datasets(output_path, dataset, dataset_full)
            result_msg = (
                f"Datasets salvos com sucesso. "
                f"Train: {train_file_path} (documentos: {exportados}). "
                f"Full: {full_file_path} (documentos: {len(dataset_full)})."
            )
            print_and_log(result_msg)
        except Exception as e:
            print_error(f"Erro ao salvar arquivo JSON: {e}")
            tmpErros += f"Erro de File IO: {e}\n"

        elapsed = time.time() - iniTime
        str_elapsed = f"Duração: {elapsed/60:.2f} min"
        print_and_log(f"Processamento export_LLM finalizado. Total avaliados: {processados}, Exportados: {exportados}. {str_elapsed}")

        if tmpErros != "":
            return {
                "status": "Processado com erros",
                "message": f"Observações: {tmpErros}. Avaliados: {processados}, Exportados: {exportados}. {str_elapsed}"
            }
        else:
            return {
                "status": "Sucesso",
                "message": f"{result_msg} Tempo decorrido: {str_elapsed}"
            }
