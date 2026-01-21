#classifica_textos_pendentesBll.py
import json
import os
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
from sqlalchemy import RowMapping, Sequence, text
from sympy import Id
import torch
from tqdm import tqdm
from sqlalchemy.orm import Session
from common import print_with_time, print_error, get_localconfig
from bll.classifica_textoBll import classifica_textoBll as classifica_textoBllModule
import bll.embeddingsBll as embeddingsBllModule
from bll.log_ClassificacaoBll import LogClassificacaoBll as LogClassificacaoBllModule
import gpu_utils as gpu_utilsModule
import logger

class ClassificaTextosPendentesBll:
    def __init__(self, session: Session):
        """
        Initialize the FoundIdIguais class for detecting similar text embeddings.
        Args:
            session (Session): SQLAlchemy session for database operations.
            localcfg: The localconfig module to read configuration.
        """
        try:                        
            from main import localconfig as localcfg # importa localconfig do main.py
            self.session = session
            self.localconfig = localcfg
            self.config = localcfg.read_config()
            self.dataset_path = Path(self.config["dataset_path"])
            self.embeddings_dir = Path(localcfg.getEmbeddingsTrain())
            self.log_dir = "../log"
            self.field = "text"
            self.k = 20  # Number of nearest neighbors to search
            self.classifica_textoBll = classifica_textoBllModule(embeddingsModule=embeddingsBllModule.bllEmbeddings,session=session)
            self.log_ClassificacaoBll = LogClassificacaoBllModule(session)
            self.logger = logger.log
            self.gpu_utils = gpu_utilsModule.GpuUtils()     
            self.limiteItensClassificar = localcfg.get("text_limit_per_batch")
            self.limitePalavras =    localcfg.get("max_length")

        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar ClassificaTextosPendentesBll: {e}")
        
    def _get_qtd_textos_pendentes(self) -> int:        
        try:
            query = f"""
                SELECT Count(t.id) AS TotalTextosPendentes
                FROM textos_classificar t    
                WHERE t.Classificado = false
                and t.TxtTreinamento IS NOT NULL
                AND t.TxtTreinamento <> ''
                and t.QtdPalavras <= {self.limitePalavras}
                ORDER BY t.id                
            """            
            return self.session.execute(text(query)).mappings().all()[0]['TotalTextosPendentes']

        except Exception as e:
            raise RuntimeError(f"Erro ontendo _get_Textos_Pendentes: {e}")            



    def _get_data_to_classify(self) -> List[RowMapping]:        
        try:
            query = f"""
                SELECT t.id,t.TxtTreinamento AS Text                   
                FROM textos_classificar t            
                WHERE t.Classificado = false
                and t.TxtTreinamento IS NOT NULL
                AND t.TxtTreinamento <> '' 
                And t.QtdPalavras <= {self.limitePalavras}               
                ORDER BY t.id
                limit {self.limiteItensClassificar}
            """
        
            return self.session.execute(text(query)).mappings().all() # pyright: ignore[reportReturnType]

        except Exception as e:
            raise RuntimeError(f"Erro ao obter dados do banco em textos_classificar: {e}")            


    def _commit_log_classificacao(self , results : list[classifica_textoBllModule.ResultadoSimilaridade] ):
        lista_log_classificacao = []
        for result in results:
            try:
                lista_log_classificacao.append({
                    "IdEncontrado": result.IdEncontrado,
                    "IdAClassificar": result.IdEncontrado,
                    "Metodo": result.Metodo,
                    "TabelaOrigem": "T"
                })            
            except Exception as e:
                print_error(f"Erro ao gravar log de classificação para Id {result.IdEncontrado}: {e}")
                continue

    def _grava_classificacao_textos_pendentes(self, itens_classificados: list[dict] ):
        BATCH_SIZE = 100
        session = self.session         
        try:           
            query = """
                Update textos_classificar set CodClasseInferido = :cod_classe_inferido , Similaridade = :similaridade, Metodo = :metodo,IdReferencia = :id_referencia,
                Classificado = 1
                where id = :id_classificado
            """         
                           
            # Process logs in chunks of BATCH_SIZE
            for i in range(0, len(itens_classificados), BATCH_SIZE):
                batch = itens_classificados[i:i + BATCH_SIZE]
                batch_params = [
                    {
                        "cod_classe_inferido": classificado["CodClasseInferido"],
                        "similaridade": (classificado["Similaridade"] or 0)*100,
                        "metodo": classificado["Metodo"],
                        "id_classificado": classificado["IdAClassificar"],                        
                        "id_referencia": classificado["IdEncontrado"] if classificado.get("IdEncontrado") is not None else None
                    }
                    for classificado in batch
                ]
                
                try:
                    # Execute batch insert
                    session.execute(text(query), batch_params)
                    session.commit()
                    self.logger.info(f"Successfully committed batch of {len(batch)} classification logs")
                except Exception as e:                    
                    print_with_time(f"Error inserting batch of {len(batch)} classification logs: {e}")
                    session.rollback()
                    continue
        except Exception as e:            
            print_with_time(f"Error processing batch classification logs: {e}")
            session.rollback()            


    #classifica os contido em textos_classificar que ainda não foram classificados
    def classifica_textos_pendentes(self) -> dict[str, Any]:
        from bll.sugere_textos_classificarBll import sugere_textos_classificarBll as sugere_textos_classificarBLLModule

        print_with_time(f"Iniciando processamento para classificação de textos a classificar pendentes...")

        embeddingsBllModule.initBllEmbeddings(session=self.session)  # inicializa bllEmbeddings se ainda não foi inicializado
                
        #Obtem dados do banco que falta classificar
        data = self._get_data_to_classify() 
        
        lista_log_classificacao = []

        for i,row in enumerate(tqdm(data, desc="Processando textos a classificar pendentes")):
            id_texto = row['id']
            texto = row['Text']            
            result = self.classifica_textoBll.classifica_texto( texto,
                                                                id_a_classificar=id_texto,
                                                                TabelaOrigem="T",
                                                                limite_itens=20)            
            lista_log_classificacao.append({
                "IdEncontrado": result.IdEncontrado,
                "IdAClassificar": id_texto,
                "Metodo": result.Metodo,
                "TabelaOrigem": "T",
                "CodClasseInferido": result.CodClasse,
                "Similaridade": result.Similaridade
            })

                # Clear cache every X batches
            if i % 20 == 0:
                self.gpu_utils.clear_gpu_cache()          
            
        
        self._grava_classificacao_textos_pendentes(lista_log_classificacao)
        self.log_ClassificacaoBll.gravaLogClassificacaoBatch(lista_log_classificacao)
    

        sucessMessage = f"Processados {len(lista_log_classificacao)} textos a classificar pendentes."
        print_with_time(sucessMessage)
        
        sugere_textos_classificarBll = sugere_textos_classificarBLLModule(self.session)                
        sugere_textos_classificarBll.sugere_textos_para_classificar()        
    
        return {
            "status": "OK",
            "processados": sucessMessage,
            "restate": f"Restam {self._get_qtd_textos_pendentes()} textos a classificar pendentes."
        }
