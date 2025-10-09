import os
from pathlib import Path
import re
from sympy import false
from typing_extensions import runtime
import numpy as np
from requests import Session
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from common import print_with_time, print_error
from collections import Counter
from bll.idsDuplicadosBll import IdsDuplicados
from sqlalchemy import text
import gpu_utils as gpu_utilsModule
import localconfig
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule
import faiss
from bll.embeddingsBll import get_bllEmbeddings


#gera embeddings dos dados contidos na tabela textos_treinamento
class Embeddings_GenerateBll:
    def __init__(self,  session: Session):
        from main import localconfig        
        self.session = session
        self.localconfig = localconfig
        self.config = localconfig.read_config()
        self.collection_name = f"v{localconfig.get('codcli')}_textos_treinamento"
        self.embeddingsTrain = localconfig.getEmbeddingsTrain()
        self.model_path = Path(self.config["model_path"])
        self.max_length = self.config["max_length"]
        self.batch_size = 200
        self.textual_fields = {'Text'}
        self.metadata_fields = {'Classe', 'Id', 'CodClasse', 'QtdItens'}
        self.tokenizer = None
        self.model = None
        self.ids_duplicados = IdsDuplicados(session)
        self.gpu_utils = gpu_utilsModule.GpuUtils()
        self.qdrant_utils = Qdrant_UtilsModule()
        self.qdrant_client = self.qdrant_utils.get_client()
        self.qdrant_utils.create_collection(self.collection_name)
        self.bllEmbeddings = get_bllEmbeddings(session)
        self.limiteItensClassificar = 5000  
        self.baseWhereSQL = """
                                WHERE LENGTH(TRIM(t.TxtTreinamento)) > 0
                                AND t.CodClasse IS NOT NULL
                                AND not t.id in (Select id from idsduplicados)
                                AND not t.id in (Select id from idsiguais)
                                and t.Indexado = false
                                and QtdPalavras <= 1024
                                and t.id in (
                                    34397493,
                                    34398205,
                                    34441817,
                                    34441916,
                                    34443568
                                    )
                            """  # Filtra textos não vazios e não nulos, não duplicados, não iguais e não indexados

        # Validate model directory
        if not os.path.isdir(self.model_path):            
            raise RuntimeError(f"Diretório do modelo não encontrado: {self.model_path}")
        
    #Valida um exemplo com base nos campos textuais e de metadados definidos.       
    def _validate_example(self, exemplo):
        if not isinstance(exemplo, dict):
            print_with_time(f"Ignorando exemplo inválido (não é um dicionário): {exemplo}")
            return False
        all_fields = self.textual_fields | self.metadata_fields

        for field in all_fields:
            if field not in exemplo:
                print_with_time(f"Ignorando exemplo com campo ausente '{field}': {exemplo}")
                return False
            value = exemplo[field]
            if field in self.textual_fields:
                if not isinstance(value, str) or not value.strip():
                    print_with_time(f"Ignorando exemplo com campo textual inválido/vazio '{field}': {exemplo}")
                    return False
            elif value is None:
                print_with_time(f"Ignorando exemplo com campo de metadados nulo '{field}': {exemplo}")
                return False
                        
        return True

    def _get_data_to_process(self) -> int:
        try:
            query = f"""
                SELECT COUNT(t.id) AS TotalTextosPendentes
                FROM textos_treinamento t    
                {self.baseWhereSQL}                
            """            
            return self.session.execute(text(query)).mappings().all()[0]['TotalTextosPendentes']

        except Exception as e:
            raise RuntimeError(f"Erro ontendo _get_Textos_Pendentes: {e}")
        

    #faz a consulta no banco de dados para obter os dados a serem processados
    def _fetch_data(self) -> list:
        # Define consulta SQL (determinística com MIN(t.id))
        query = f"""
            SELECT MIN(t.id) AS Id,
                   c.CodClasse,
                   c.Classe,
                   t.TxtTreinamento AS Text,
                   COUNT(t.id) AS QtdItens
            FROM textos_treinamento t
            INNER JOIN classes c ON c.CodClasse = t.CodClasse
            {self.baseWhereSQL}
            GROUP BY t.TxtTreinamento, t.CodClasse, c.Classe
            Order by COUNT(t.id),t.id DESC
            limit {self.limiteItensClassificar}
        """
        # Busca dados do banco de dados
        try:
            result = self.session.execute(text(query)).mappings().all()
            dados = [dict(row) for row in result]

            if not dados:
                raise RuntimeError("Não há dados para gerar o modelo de embeddings.")
            
            return dados
        except Exception as e:
            raise RuntimeError(f"Erro executando consulta no banco de dados: {e}")
        
    #insere os dados processados no Qdrant    
    def _insert_data_in_qdrant(self, processed_data: list[dict]) -> tuple[str,int]:
        tmpError = ""        
        try:

            if not processed_data:
                tmpError = "Sem dados para processar no lote."
                print_with_time(tmpError)
                return tmpError,0

            #verificar se a estrutura dos dados processados está correta    
            for item in processed_data:
                if 'Embedding' not in item or 'Metadata' not in item:
                    tmp = f"Item processado incompleto: {item}"
                    tmpError += tmp + "\n"
                    raise RuntimeError(tmp)
                
                for field in self.metadata_fields:
                    if field not in item['Metadata']:
                        tmp = f"Item processado com metadado ausente '{field}': {item}"
                        tmpError += tmp + "\n"    
                        raise RuntimeError(tmp)

            
            print_with_time(f"Salvando {len(processed_data)} embeddings no Qdrant...")
            processed_data = self._insert_lista_texto_qdrant(processed_data)

            idsToUpdate = []
            for item in processed_data:
                if item.get('UpInsertOk') == False:
                    tmp = f"Falha ao inserir no banco embedding para ID {item.get('Id')}"
                    print_with_time(tmp)
                    tmpError += tmp + "\n"
                else:
                    idsToUpdate.append(item.get('Id'))


            if idsToUpdate:
                tmpError += self._update_indexados_in_ids(idsToUpdate)#atualiza os ids que foram processados com sucesso


            return tmpError,len(idsToUpdate) # tudo certo retorna string vazia
        
        except Exception as e:
            tmp = f"Erro ao inserir dados no Qdrant: {e}"
            print_with_time(tmp)
            tmpError += tmp + "\n"
            return tmpError,0
        
    #atualiza os ids que foram processados com sucesso
    def _update_indexados_in_ids(self,ids: list[int]) -> str:
            tmpError = ""
            try:
                # Atualiza o campo indexado para os IDs inseridos com sucesso
                update_query = text(f"""
                    UPDATE textos_treinamento
                    SET Indexado = true
                    WHERE id in :ids
                """)
                self.session.execute(update_query, {"ids": ids})
                self.session.commit()                
            except Exception as e:
                tmp = f"Erro ao atualizar campo 'indexado' no banco de dados: {e}"
                print_with_time(tmp)
                tmpError += tmp + "\n"
            
            return tmpError # tudo certo retorna string vazia
    

    #Processa um lote de dados para gerar embeddings gerar uma lista e inserir no Qdrant
    def _process_data(self, batch_dados: list) -> tuple[str,int]:
        tmpErrors = ""
        print_with_time(f"Memória inicial da GPU: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        processed_data = []
        skipped = 0
        invalid_reasons = Counter()

        print_with_time(f"Processando lote com {len(batch_dados)} registros ...")

        # Processa cada exemplo individualmente
        for exemplo in tqdm(batch_dados, desc="Gerando embeddings individuais"):
            if not self._validate_example(exemplo):
                skipped += 1
                invalid_reasons['invalid_example'] += 1
                continue

            try:
                # Gera embedding para o texto individual
                text = exemplo['Text']
                embedding = self.bllEmbeddings.generate_embedding(text, None)  # Processa um texto por vez

                # Armazena embedding e metadados
                processed_data.append({
                    'Id': exemplo['Id'],
                    'Embedding': embedding,
                    'UpInsertOk': False,
                    'Metadata': {f: exemplo[f] for f in self.metadata_fields}
                })

                # Verifica duplicatas
                if exemplo['QtdItens'] > 1:
                    self.ids_duplicados.insert_duplicate_ids(
                        id=exemplo['Id'],
                        texto=exemplo['Text'],
                        cod_classe=exemplo['CodClasse']
                    )

            except Exception as e:
                tmp = f"Erro ao processar exemplo {exemplo['Id']}: {e}"
                print_with_time(tmp)
                tmpErrors += tmp + "\n"
                skipped += 1
                continue

        # Relatório do lote
        print_with_time(f"Lote: {len(batch_dados) - skipped} exemplos processados, {skipped} exemplos ignorados.")
        if invalid_reasons:
            tmpErrors += f"Motivos de ignorados: {invalid_reasons}"
            print_with_time(tmpErrors)

        if (skipped > 0):
            tmpErrors += f" Aviso: {skipped} exemplos foram ignorados neste lote.\n"

        result = self._insert_data_in_qdrant(processed_data)
        tmpErrors += result[0]

        if tmpErrors != "":
            return tmpErrors,result[1]
        else:
            return "",result[1] # se tiver tudo certo retorna string vazia
        
        
    
    ###Insere uma lista de embeddings e metadados no Qdrant.
    def _insert_lista_texto_qdrant(self, processed_data: list[dict]) -> list[dict]:
        try:
            tmp = ""
            points = []
            for item in tqdm(processed_data, desc="Preparando pontos para Qdrant"):
                embedding = item['Embedding'].astype('float32')
                faiss.normalize_L2(embedding)
                points.append(PointStruct(
                    id=item['Id'],
                    vector=embedding.flatten().tolist(),
                    payload={
                        "id": item['Id'],
                        "Classe": item['Metadata']['Classe'],
                        "CodClasse": item['Metadata']['CodClasse'],
                        "QtdItens": item['Metadata']['QtdItens']
                    }
                ))

            result = self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            if result.status == "completed":
                for item in processed_data:
                    item['UpInsertOk'] = True

                print_with_time(f"Inseridos {len(processed_data)} textos no Qdrant com sucesso.")
            else:
                for item in processed_data:
                    item['UpInsertOk'] = False

                tmp = f"Falha ao inserir textos no Qdrant: status {result.status}"
                print_with_time(tmp)

            return processed_data
        
        except Exception as e:
            print_with_time(f"Erro ao inserir lista de textos no Qdrant: {e}")
            for item in processed_data:
                item['UpInsertOk'] = False

            return processed_data

    #Inicia o processo de geração de embeddings
    def start(self):
        dados = self._fetch_data()
        print_with_time(f"Total de registros a processar: {len(dados)}")
        
        tmpErros = ""
        processados = 0
        # Divide os dados em lotes
        for i in tqdm(range(0, len(dados), self.batch_size), desc="Processando lotes"):
            batch_dados = dados[i:i + self.batch_size]
            result = self._process_data(batch_dados)
            tmpErros += result[0]
            processados += result[1]
        
        if tmpErros != "":
            return {"status": "Processado com erros",
                    "message": f"Erros {tmpErros} faltam processar {self._get_data_to_process()} registros. processados {processados} registros."}
        else:
            return {"status": "Sucesso",
                    "message": f"Faltam processar {self._get_data_to_process()} registros."}