#LogClassificacaoBll.py
from typing import Optional
from sqlalchemy import text
from db_utils import Session
from common import print_with_time, print_error
import logger
from datetime import datetime

class LogClassificacaoBll:
    def __init__(self, session: Session):
        self.session = session
        self.logger = logger.log

    def gravaLogClassificacao(self, id_referencia: int, id_classificado:  int, metodo: str, tabela_origem: str, cod_classe_inferido: Optional[int]):
        """
        Inserts a classification log entry into the log_classificacao table.

        Args:
            id_referencia (int): The reference ID (IdReferencia).
            id_classificado (int): The classified ID (IdClassificado), can be None for 'A' (Avulso).
            metodo (str): The classification method used.
            tabela_origem (str): The origin of the text ('C', 'T', or 'A').
        """
        session = self.session        
        try:
            query = """
                INSERT INTO log_classificacao (IdReferencia, IdClassificado, DataHora, TabelaOrigem, Metodo, CodClasseInferido )
                VALUES (:id_referencia, :id_classificado, :data_hora, :tabela_origem, :metodo, :cod_classe_inferido)
            """
            session.execute(
                text(query),
                {
                    "id_referencia": id_referencia,
                    "id_classificado": id_classificado,
                    "data_hora": datetime.now(),
                    "tabela_origem": tabela_origem,
                    "metodo": metodo,
                    "cod_classe_inferido": cod_classe_inferido
                }
            )
            session.commit()          
        except Exception as e:
            self.logger.error(f"Error inserting classification log (IdReferencia: {id_referencia}): {e}")
            print_error(f"Error inserting classification log (IdReferencia: {id_referencia}): {e}")
            session.rollback()

    #   Inserts multiple classification log entries into the log_classificacao table in batches of 100,
    #    committing each batch and any remaining records.
    def gravaLogClassificacaoBatch(self, logs: list[dict]):        
        session = self.session            
        BATCH_SIZE = 100
        try:        
            query = """
                INSERT ignore INTO  log_classificacao (IdReferencia, IdClassificado, DataHora, TabelaOrigem, Metodo,CodClasseInferido)
                VALUES (:id_referencia, :id_classificado, :data_hora, :tabela_origem, :metodo, :cod_classe_inferido)
            """
            
            # Process logs in chunks of BATCH_SIZE
            for i in range(0, len(logs), BATCH_SIZE):
                batch = logs[i:i + BATCH_SIZE]
                batch_params = [
                    {
                        "id_referencia": log["IdEncontrado"],
                        "id_classificado": log["IdAClassificar"],
                        "data_hora": datetime.now(),
                        "tabela_origem": log["TabelaOrigem"],
                        "metodo": log["Metodo"],
                        "cod_classe_inferido": log["CodClasseInferido"]
                    }
                    for log in batch
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