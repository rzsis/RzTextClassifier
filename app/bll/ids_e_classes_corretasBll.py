from sqlalchemy import text
from db_utils import Session
from common import print_with_time
import numpy as np
import logger

class IdsEClassesCorretasBll:
    def __init__(self, session: Session):
        self.session = session
        self.logger = logger.log

    def _get_ids_e_Classes_Corretas(self) -> list[dict[str, any]]:
        try:
            query = """
                SELECT t.id AS Id, t.CodClasse AS CodClasse, c.Classe AS Classe
                FROM ideclassecorreta t
                INNER JOIN classes c ON c.CodClasse = t.CodClasse
                
            """
            rows = self.session.execute(text(query)).mappings().all()
            return rows
        except Exception as e:
            error = f"Erro obtendo get_ids_e_Classes_Corretas: {e}"
            self.logger.error(error)
            print_with_time(error)
            return []

    def corrige_metadata(self, metadata: np.lib.npyio.NpzFile) -> dict:
        """
        Cria uma cópia do metadata, atualiza CodClasse e Classe apenas para registros cujo Id está em
        ideclassecorreta, e preserva a ordem original dos índices para manter a correspondência com os embeddings.

        Args:
            metadata (np.lib.npyio.NpzFile): Metadados carregados contendo Id, CodClasse, Classe, QtdItens.

        Returns:
            dict: Novo dicionário com a mesma estrutura e ordem de índices do metadata original, com
                  CodClasse e Classe atualizados para IDs em ideclassecorreta.

        Raises:
            RuntimeError: Se faltarem campos esperados no metadata ou ocorrer um erro durante a correção.
        """
        try:
            # Verificar se os campos esperados estão no metadata
            expected_fields = {'Id', 'CodClasse', 'Classe', 'QtdItens'}
            if not all(field in metadata for field in expected_fields):
                missing = expected_fields - set(metadata.keys())
                raise RuntimeError(f"Campos faltando no metadata: {missing}")

            # Obter os dados corretos do banco de dados (tabela ideclassecorreta)
            dados_corretos = self._get_ids_e_Classes_Corretas()
            
            # Criar dicionário para mapear Id para CodClasse e Classe corretos
            id_to_correct_data = {row['Id']: {'CodClasse': row['CodClasse'], 'Classe': row['Classe']} for row in dados_corretos}
            
            # Criar cópias dos arrays de metadata para evitar modificar o original e preservar a ordem dos índices
            ids = metadata['Id'].copy()
            cod_classes = metadata['CodClasse'].astype(int).copy()  # Garantir tipo int
            classes = metadata['Classe'].copy()
            qtd_itens = metadata['QtdItens'].copy()
            
            updated_count = 0
            # Iterar sobre os índices na ordem original para atualizar apenas registros com Id em ideclassecorreta
            for i, id_ in enumerate(ids):
                if id_ in id_to_correct_data:
                    correct_data = id_to_correct_data[id_]
                    # Atualizar apenas se os valores forem diferentes, mantendo o índice i
                    if cod_classes[i] != correct_data['CodClasse'] or classes[i] != correct_data['Classe']:
                        cod_classes[i] = correct_data['CodClasse']
                        classes[i] = correct_data['Classe']
                        updated_count += 1
                    # Dados originais são mantidos para IDs não encontrados em id_to_correct_data
            
            print_with_time(f"Metadados corrigidos: {updated_count} registros atualizados com CodClasse e Classe de ideclassecorreta.")
            
            # Retornar novo dicionário com a mesma estrutura e ordem de índices do metadata original
            updated_metadata = {
                'Id': ids,
                'CodClasse': cod_classes,
                'Classe': classes,
                'QtdItens': qtd_itens
            }
            
            return updated_metadata
        
        except Exception as e:
            print_with_time(f"Erro ao corrigir metadados: {e}")
            raise RuntimeError(f"Erro ao corrigir metadados: {e}")