    
from db_utils import Session
from threading import Lock
from typing import Optional
from sqlalchemy import text

classes_utils_singleton = None


def initClassesUtils(session: Session):
    global classes_utils_singleton
    if classes_utils_singleton is None:
        with Lock():
            if classes_utils_singleton is None:
                classes_utils_singleton = classes_utilsBLL(session=session)


def get_ClassesUtils() -> 'classes_utilsBLL':
    if classes_utils_singleton is None:
        raise RuntimeError("classes_utilsBLL não inicializado. Chame initClassesUtils(session) primeiro.")
    return classes_utils_singleton  


# Para otimizar a busca de classes em memoria ao invés de consulta o banco foi criada a classe classes_utilsBLL que carrega as classes em memoria e tem métodos para acessar as informações das classes
class classes_utilsBLL:
        def __init__(self, session: Session):
            self.session = session
            self._classes: list[dict] = []
            self._classes_by_cod: dict[int, str] = {}
            self.atualizar_lista_classes()

        def _carregar_classes(self) -> list[dict]:
            query = """
                SELECT
                    c.CodClasse,
                    c.CodSubClasse,
                    c.Classe,
                    c.CodClasseTreinamento
                FROM classes c
                ORDER BY c.CodClasse
            """
            return [dict(row) for row in self.session.execute(text(query)).mappings().all()]

        def atualizar_lista_classes(self) -> None:
            try:
                self._classes = self._carregar_classes()
                self._classes_by_cod = {
                    int(item["CodClasse"]): str(item["Classe"])
                    for item in self._classes
                    if item.get("CodClasse") is not None and item.get("Classe") is not None
                }
            except Exception as e:
                raise RuntimeError(f"Erro ao atualizar lista de classes: {e}")

        def get_lista_classes(self) -> list[dict]:
            return list(self._classes)

        def get_nome_classe(self, codclasse: int) -> str | None:
            return self._classes_by_cod.get(int(codclasse))