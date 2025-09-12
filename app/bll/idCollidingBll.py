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