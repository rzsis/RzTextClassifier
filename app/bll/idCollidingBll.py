from sqlalchemy.orm import Session
from common import print_error
from sqlalchemy import text

class IdCollidingBll:
    def __init__(self, session: Session):
        self.session = session

    def commit_lista(self, objetos_idscolidentes):
        if not objetos_idscolidentes:
            return 0
        try:
            self.session.add_all(objetos_idscolidentes)
            self.session.commit()
            return len(objetos_idscolidentes)
        except Exception as e:
            self.session.rollback()
            print_error(f"[ERRO] Falha ao inserir idscolidentes: {e}")
            raise

    def limpa_registros(self):
        """Limpa todos os registros da tabela idscolidentes."""
        try:
            self.session.execute(text("DELETE FROM idscolidentes"))
            self.session.commit()
            print("Todos os registros de idscolidentes foram removidos com sucesso.")
        except Exception as e:
            self.session.rollback()
            print_error(f"[ERRO] Falha ao limpar registros de idscolidentes: {e}")
            raise

    #Retorna todos os registros da tabela idscolidentes.
    #gerando uma lista dupla pois uma vez inserido o Id colidente como base ou similar ele não deve ser inserido denovo
    def get_all_ids_colidentes(self):    
        try:
            query = """
                    SELECT IdColidente FROM idscolidentes
                    Union
                    SELECT Id FROM idscolidentes 
                    """
            result = self.session.execute(text(query))
            rows = result.mappings().all()
            return {row["IdColidente"] for row in rows}        
                    
        except Exception as e:
            print_error(f"[ERRO] Falha ao buscar registros de idscolidentes: {e}")
            raise

    def set_buscou_colidente(self, id_list):
        """Marca os registros como buscou colidente."""
        if not id_list:
            return
        
        try:
            query = text("""
                UPDATE textos_treinamento
                SET BuscouColidente = true
                WHERE Id IN :id_list
            """)
            self.session.execute(query, {"id_list": tuple(id_list)})
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print_error(f"[ERRO] Falha ao atualizar BuscouColidente: {e}")
            raise