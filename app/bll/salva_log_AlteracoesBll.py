from db_utils import Session
from sqlalchemy import RowMapping, Sequence, text

#grava log de alterações nas tabelas monitoradas
class salva_log_AlteracoesBll:
    def __init__(self, session: Session):
        self.session = session
        self.tabelas = []
        self.tabelas.append({
            'tabela': 'textos_treinamento',
            'codTabela': self._get_database_cod_Tabela('textos_treinamento')
        })
    
    #obtem o codTabela pelo nome da tabela na lista de tabelas que deve ser montada previamente 
    def _get_cod_tabela_by_name(self, tabela_name: str) -> int:
        for tabela in self.tabelas:
            if tabela['tabela'] == tabela_name:
                return tabela['codTabela']
            
        raise RuntimeError(f"Tabela '{tabela_name}' não encontrada na lista de tabelas.")

    #obtem o codTabela pelo nome da tabela salva em banco
    def _get_database_cod_Tabela(self, tabela: str) -> int:
        try:
            sql = text(f"SELECT CodTabela FROM tabelas WHERE tabela = :tabela")
            result = self.session.execute(sql, {'tabela': tabela}).fetchone()
            if result:
                return result[0]
            else:
                raise RuntimeError(f"Tabela '{tabela}' não encontrada.")
            
        except Exception as e:
            raise e

    #insere log de alterações para textos_treinamento
    def insert_log_texto_treinamento(self, ids:list[int], codUser:int, auto_commit:bool=False):
        try:
            codtabela = self._get_cod_tabela_by_name('textos_treinamento')
            for id in ids:
                sql = text(f"""
                    INSERT INTO usuarioslogalteracoes (`Key`, CodUser, TipoAlteracao, DataHora, CodTabela)
                    VALUES (:key, :cod_user, :tipo_alteracao, CURRENT_TIMESTAMP, :CodTabela)
                """)
                self.session.execute(sql, {
                    'key': id,                  
                    'cod_user': codUser, 
                    'tipo_alteracao': 'I',              
                    'CodTabela': codtabela,                                
                })
            
            
            if auto_commit:
                self.session.commit()

        except Exception as e:
            if auto_commit:
                self.session.rollback()                

            raise RuntimeError(f"Erro ao inserir log de alterações para textos_treinamento: {str(e)}")