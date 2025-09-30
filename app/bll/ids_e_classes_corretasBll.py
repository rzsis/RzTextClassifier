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
        Atualiza os campos CodClasse e Classe no metadata com base nos dados corretos do banco de dados.

        Args:
            metadata (np.lib.npyio.NpzFile): Metadados carregados contendo Id, CodClasse, Classe, etc.

        Returns:
            dict: Metadados atualizados como dicionário.

        Raises:
            RuntimeError: Se ocorrer um erro durante a correção dos metadados.
        """
        try:
            # Obter os dados corretos do banco de dados
            dados_corretos = self._get_ids_e_Classes_Corretas()
            
            # Criar um dicionário para mapear Id para CodClasse e Classe corretos
            id_to_correct_data = {row['Id']: {'CodClasse': row['CodClasse'], 'Classe': row['Classe']} for row in dados_corretos}
            
            # Copiar os arrays de metadata para modificação
            ids = metadata['Id']
            cod_classes = metadata['CodClasse'].astype(int)
            classes = metadata['Classe']
            qtd_itens = metadata['QtdItens']  # Preservar outros campos, como QtdItens
            
            updated_count = 0
            # Atualizar CodClasse e Classe onde houver correspondência
            for i, id_ in enumerate(ids):
                if id_ in id_to_correct_data:
                    correct_data = id_to_correct_data[id_]
                    if cod_classes[i] != correct_data['CodClasse'] or classes[i] != correct_data['Classe']:
                        cod_classes[i] = correct_data['CodClasse']
                        classes[i] = correct_data['Classe']
                        updated_count += 1
            
            print_with_time(f"Metadados corrigidos: {updated_count} registros atualizados com CodClasse e Classe corretos.")
            
            # Criar um novo dicionário para os metadados atualizados
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