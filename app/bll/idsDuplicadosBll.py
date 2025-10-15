# db_idsduplicados.py
from sqlalchemy import text
from db_utils import Session
from common import print_with_time, print_error
import logger

class IdsDuplicados:
    def __init__(self, db: Session):
        self.db = db
        self.logger = logger.log

    # Insere IDs duplicados na tabela idsduplicados
    def insert_duplicate_ids(self, idBase: int, texto: str, cod_classe: int):
        session = self.db
        try:
            
            query = f""" 
                INSERT ignore INTO  idsduplicados (Id, IdDuplicado,CodClasse)
                SELECT :id_base, t.Id, :cod_classe
                FROM textos_treinamento t
                WHERE t.TxtTreinamento = :texto
                AND t.Id <> :id
                AND t.Id NOT IN (SELECT Id FROM idsduplicados)
                and t.CodClasse = :cod_classe
            """
            session.execute(
                text(query),
                {"id_base":idBase, "cod_classe": cod_classe, "texto": texto, "id": idBase}
            )
            session.commit()            
        except Exception as e:
            self.logger.error(f"Error inserting duplicate text (ID: {idBase}): {e}")
            print_error(f"Error inserting duplicate text (ID: {idBase}): {e}")
            session.rollback()
        