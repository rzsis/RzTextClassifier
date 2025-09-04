# db_idsduplicados.py
from sqlalchemy import text
from db_utils import Db
from common import print_with_time, print_error
import logger

class IdsDuplicados:
    def __init__(self, db: Db):
        self.db = db
        self.logger = logger.log

    def insert_duplicate_ids(self, id: int, texto: str, cod_classe: int):
        """
        Inserts duplicate IDs into the idsduplicados table for a given text and CodClasse.
        
        Args:
            id (int): The reference ID to exclude from duplicates.
            texto (str): The text to check for duplicates (TxtTreinamento).
            cod_classe (int): The CodClasse associated with the text.
        """
        try:
            session = self.db
            query = """
                INSERT ignore INTO  idsduplicados (Id, CodClasse)
                SELECT t.Id, :cod_classe
                FROM textos_treinamento t
                WHERE t.TxtTreinamento = :texto
                AND t.Id <> :id
                AND t.Id NOT IN (SELECT Id FROM idsduplicados)
            """
            session.execute(
                text(query),
                {"cod_classe": cod_classe, "texto": texto, "id": id}
            )
            session.commit()
            print_with_time(f"Inserted duplicate IDs for text (ID: {id}, CodClasse: {cod_classe})")
        except Exception as e:
            self.logger.error(f"Error inserting duplicate text (ID: {id}): {e}")
            print_error(f"Error inserting duplicate text (ID: {id}): {e}")
            session.rollback()
        