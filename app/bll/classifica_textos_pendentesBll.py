#classifica_textos_pendentesBll.py
import os
from pathlib import Path
import numpy as np
import faiss
from sqlalchemy import text
from sympy import Id
from tqdm import tqdm
from sqlalchemy.orm import Session
from common import print_with_time, print_error, get_localconfig
from bll.classifica_textoBll import classifica_textoBll as classifica_textoBllModule
import bll.embeddingsBll as embeddingsBllModule
from bll.log_ClassificacaoBll import LogClassificacaoBll as LogClassificacaoBllModule
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
            self.classifica_textoBll = classifica_textoBllModule(embeddingsModule=embeddingsBllModule.bllEmbeddings,
                            session=session)
            self.log_ClassificacaoBll = LogClassificacaoBllModule(session)
            self.logger = logger.log            



        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar ClassificaTextosPendentesBll: {e}")
        
    def _get_Textos_Pendentes(self) -> list:        
        try:
            query = """
                SELECT Count(t.id) AS TotalTextosPendentes
                FROM textos_classificar t    
                WHERE t.Classificado = false
                and t.TxtTreinamento IS NOT NULL
                AND t.TxtTreinamento <> ''
                and Classificado = false
                ORDER BY t.id                
            """
        
            return self.session.execute(text(query)).mappings().all()

        except Exception as e:
            raise RuntimeError(f"Erro ontendo _get_Textos_Pendentes: {e}")            



    def _fetch_data(self) -> list:        
        try:
            query = """
                SELECT t.id,t.TxtTreinamento AS Text                   
                FROM textos_classificar t            
                WHERE t.Classificado = false
                and t.TxtTreinamento IS NOT NULL
                AND t.TxtTreinamento <> ''
                and Classificado = false
                ORDER BY t.id
                limit 2000
            """
        
            return self.session.execute(text(query)).mappings().all()

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
        try:
            session = self.session            
            query = """
                Update textos_classificar set CodClasseInferido = :cod_classe_inferido , Similaridade = :similaridade, Metodo = :metodo,IdReferencia = :id_referencia,
                Classificado = true
                where id = :id_classificado
            """         
                           
            # Process logs in chunks of BATCH_SIZE
            for i in range(0, len(itens_classificados), BATCH_SIZE):
                batch = itens_classificados[i:i + BATCH_SIZE]
                batch_params = [
                    {
                        "cod_classe_inferido": classificado["CodClasseInferido"],
                        "similaridade": classificado["Similaridade"],
                        "metodo": classificado["Metodo"],
                        "id_classificado": classificado["IdAClassificar"],                        
                        "id_referencia": classificado["IdEncontrado"]                            
                    }
                    for classificado in batch
                ]
                
                try:
                    # Execute batch insert
                    session.execute(text(query), batch_params)
                    session.commit()
                    self.logger.info(f"Successfully committed batch of {len(batch)} classification logs")
                except Exception as e:
                    self.logger.error(f"Error inserting batch of {len(batch)} classification logs: {e}")
                    print_error(f"Error inserting batch of {len(batch)} classification logs: {e}")
                    session.rollback()
                    continue
        except Exception as e:
            self.logger.error(f"Error processing batch classification logs: {e}")
            print_error(f"Error processing batch classification logs: {e}")
            session.rollback()            


    def classifica_textos_pendentes(self) -> list[classifica_textoBllModule.ResultadoSimilaridade]:
        """
        Start processing the dataset.
        """
        print_with_time(f"Iniciando processamento para classificação de textos a classificar pendentes...")

        embeddingsBllModule.initBllEmbeddings(self.session)  # inicializa bllEmbeddings se ainda não foi inicializado
        
        # Load data from database
        data = self._fetch_data()
        
        lista_log_classificacao = []

        for row in tqdm(data, desc="Processando textos a classificar pendentes"):
            id_texto = row['id']
            texto = row['Text']            
            result = self.classifica_textoBll.classifica_texto( texto,
                                                                id_a_classificar=id_texto,
                                                                TabelaOrigem="T",
                                                                top_k=20)            
            lista_log_classificacao.append({
                "IdEncontrado": result.IdEncontrado,
                "IdAClassificar": id_texto,
                "Metodo": result.Metodo,
                "TabelaOrigem": "T",
                "CodClasseInferido": result.CodClasse,
                "Similaridade": result.Similaridade
            })
            
        
        self._grava_classificacao_textos_pendentes(lista_log_classificacao)

        self.log_ClassificacaoBll.gravaLogClassificacaoBatch(lista_log_classificacao)

        sucessMessage = f"Processados {len(lista_log_classificacao)} textos a classificar pendentes."
        print_with_time(sucessMessage)
    
        return {
            "status": "OK",
            "processados": sucessMessage,
            "restate": f"Restam {self._get_Textos_Pendentes()[0]['TotalTextosPendentes'] } textos a classificar pendentes."
        }
