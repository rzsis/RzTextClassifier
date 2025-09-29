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



        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar ClassificaTextosPendentesBll: {e}")
        
        
    def _fetch_data(self) -> list:        
        try:
            query = """
                SELECT t.id,t.TxtTreinamento AS Text                   
                FROM textos_classificar t            
                WHERE t.Classificado = false
                and t.TxtTreinamento IS NOT NULL
                AND t.TxtTreinamento <> ''
                ORDER BY t.id
                limit 10
            """
        
            return self.session.execute(text(query)).mappings().all()

        except Exception as e:
            raise RuntimeError(f"Erro ao obter dados do banco em textos_classificar: {e}")            


    def commit_log_classificacao(self , results : list[classifica_textoBllModule.ResultadoSimilaridade] ):
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


    def classifica_textos_pendentes(self) -> list[classifica_textoBllModule.ResultadoSimilaridade]:
        """
        Start processing the dataset.
        """
        print_with_time(f"Iniciando processamento para classificação de textos a classificar pendentes...")

        embeddingsBllModule.initBllEmbeddings()  # inicializa bllEmbeddings se ainda não foi inicializado
        
        # Load data from database
        data = self._fetch_data()

        lista_resultado_similaridade : list[classifica_textoBllModule.ResultadoSimilaridade] = []
        lista_log_classificacao = []

        for row in tqdm(data, desc="Processando textos a classificar pendentes"):
            id_texto = row['id']
            texto = row['Text']            
            result = self.classifica_textoBll.classifica_texto( texto,
                                                                id_a_classificar=id_texto,
                                                                TabelaOrigem="T",
                                                                top_k=20)
            lista_resultado_similaridade.append(result)
            lista_log_classificacao.append({
                "IdEncontrado": result.IdEncontrado,
                "IdAClassificar": id_texto,
                "Metodo": result.Metodo,
                "TabelaOrigem": "T"
            })
            
        
        self.log_ClassificacaoBll.gravaLogClassificacaoBatch(lista_log_classificacao)

        print_with_time(f"Processados {len(lista_resultado_similaridade)} textos a classificar pendentes concluído.")
    
        return lista_resultado_similaridade
