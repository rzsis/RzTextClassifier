import datetime
import os
from pathlib import Path
import re
import time
from sympy import false
from typing_extensions import runtime
import numpy as np
from requests import Session
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from common import print_with_time, print_error
from collections import Counter
from sqlalchemy import text
import gpu_utils as gpu_utilsModule
import localconfig
from bll.embeddingsBll import get_bllEmbeddings


#gera embeddings dos dados contidos na tabela textos_treinamento
class finetuning_bge_m3:
    def __init__(self,  session: Session):     
        self.session = session
        self.localconfig = localconfig
        self.config = localconfig.read_config()
        self.embeddingsTrain = localconfig.getEmbeddingsTrain()
        self.model_path = Path(self.config["model_path"])
        self.max_length = self.config["max_length"]                
        self.limiteItensClassificar = localconfig.get("text_limit_per_batch")

        # Validate model directory
        if not os.path.isdir(self.model_path):            
            raise RuntimeError(f"Diretório do modelo não encontrado: {self.model_path}")
        

    #faz a consulta no banco de dados para obter os dados a serem processados
    def _fetch_data(self) -> list:        
        query = f"""
            SELECT MIN(t.id) AS Id,                   
                   t.TxtTreinamento AS Text                   
            FROM textos_classificar t
                    INNER JOIN classes c ON c.CodClasse = t.CodClasse
                                                   WHERE LENGTH(TRIM(t.TxtTreinamento)) > 0                                
                                and t.Indexado = false
                                and QtdPalavras <= 512        
            GROUP BY t.TxtTreinamento, t.CodClasse, c.Classe
            Order by COUNT(t.id) DESC
            limit {self.limiteItensClassificar}
        """

        # Busca dados do banco de dados
        try:
            result = self.session.execute(text(query)).mappings().all()
            dados = [dict(row) for row in result]
                        
            return dados
        except Exception as e:
            raise RuntimeError(f"Erro executando consulta no banco de dados: {e}")
        
    
    #Inicia o processo de geração de embeddings
    def start(self):
        iniTime = time.time()  
        print_with_time(f"Iniciando processamento de finetuing em {self.model_path}... : {iniTime}")
        dados = self._fetch_data()
        qtdreg = len(dados)
        if qtdreg == 0:
            return {"status": "Completo",
                    "message": f"Não há dados para processar, todos dos textos para treinamento já foram indexados."}
    
        print_with_time(f"Total de registros a processar: {len(dados)}")
        tmpErros = ""
        processados = 0
        qtdLotes = (qtdreg // self.batch_size) + (1 if qtdreg % self.batch_size > 0 else 0)

        # Divide os dados em lotes
        indexlote = 1
        for i in tqdm(range(0, qtdreg, self.batch_size), desc=f"Processando lote"):
            batch_dados = dados[i:i + self.batch_size]
            result = self._process_data(batch_dados,indexlote,qtdLotes)
            tmpErros += result[0]
            processados += result[1]
            indexlote += 1
        
        elapsed     = time.time() - iniTime
        str_elapsed = f"Duração: {elapsed/60:.2f} min"
        print_with_time(f"Processamento finalizado. Total processado: {processados}. {str_elapsed}")

        if tmpErros != "":
            return {"status": "Processado com erros",
                    "message": f"Erros {tmpErros} faltam processar {self._get_data_to_process()} registros. processados {processados} registros, {str_elapsed}"}
        else:
            return {"status": "Sucesso",
                    "message": f"Faltam processar {self._get_data_to_process()} registros,  {str_elapsed}"}