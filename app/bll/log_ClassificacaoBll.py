#LogClassificacaoBll.py
from sqlalchemy import text
from db_utils import Db
from common import print_with_time, print_error
import logger
from datetime import datetime

class LogClassificacaoBll:
    def __init__(self, db: Db):
        self.db = db
        self.logger = logger.log

    def gravaLogClassificacao(self, id_referencia: int, id_classificado: int, metodo: str, tabela_origem: str):
        """
        Inserts a classification log entry into the log_classificacao table.

        Args:
            id_referencia (int): The reference ID (IdReferencia).
            id_classificado (int): The classified ID (IdClassificado), can be None for 'A' (Avulso).
            metodo (str): The classification method used.
            tabela_origem (str): The origin of the text ('C', 'T', or 'A').
        """
        try:
            session = self.db
            query = """
                INSERT INTO log_classificacao (IdReferencia, IdClassificado, DataHora, TabelaOrigem, Metodo)
                VALUES (:id_referencia, :id_classificado, :data_hora, :tabela_origem, :metodo)
            """
            session.execute(
                text(query),
                {
                    "id_referencia": id_referencia,
                    "id_classificado": id_classificado,
                    "data_hora": datetime.now(),
                    "tabela_origem": tabela_origem,
                    "metodo": metodo
                }
            )
            session.commit()          
        except Exception as e:
            self.logger.error(f"Error inserting classification log (IdReferencia: {id_referencia}): {e}")
            print_error(f"Error inserting classification log (IdReferencia: {id_referencia}): {e}")
            session.rollback()