from sqlalchemy.orm import Session
from common import print_error
from sqlalchemy import text

class IdIguaisBll:
    def __init__(self, session: Session):
        self.session = session

    def commit_lista_ids_iguais(self, objetos_idsiguais):
        if not objetos_idsiguais:
            return 0
        try:
            insertSql = text(f"""
                INSERT IGNORE INTO idsiguais (Id, IdIgual)
                VALUES (:Id, :IdIgual)
            """)

            parametros = [{'Id': item.id, 'IdIgual': item.idIgual} for item in objetos_idsiguais]
            
            self.session.execute(insertSql, parametros)                                             
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

    #atualiza o campo BuscouIgual na tabela textos_treinamento
    def set_buscou_igual(self, id_list):
        """Marca os registros como buscou igual."""
        if not id_list:
            return
        
        try:
            query = text("""
                UPDATE textos_treinamento
                SET BuscouIgual = true
                WHERE Id IN :id_list
            """)
            self.session.execute(query, {"id_list": tuple(id_list)})
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print_error(f"[ERRO] Falha ao atualizar BuscouIgual: {e}")
            raise