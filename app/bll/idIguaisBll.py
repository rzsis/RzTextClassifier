from sqlalchemy.orm import Session
from common import print_error
from sqlalchemy import text

class IdIguaisBll:
    def __init__(self, session: Session):
        self.session = session

    def commit_lista(self, objetos_idsiguais):
        if not objetos_idsiguais:
            return 0
        try:
            self.session.add_all(objetos_idsiguais)
            self.session.commit()
            return len(objetos_idsiguais)
        except Exception as e:
            self.session.rollback()
            print_error(f"[ERRO] Falha ao inserir idsiguais: {e}")
            raise

    def limpa_registros(self):
        """Limpa todos os registros da tabela idsiguais."""
        try:
            self.session.execute(text("DELETE FROM idsiguais"))
            self.session.commit()
            print("Todos os registros de IdsIguais foram removidos com sucesso.")
        except Exception as e:
            self.session.rollback()
            print_error(f"[ERRO] Falha ao limpar registros de IdsIguais: {e}")
            raise