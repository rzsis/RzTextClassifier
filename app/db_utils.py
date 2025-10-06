# db_utils.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import localconfig

class Session:
    def __init__(self,localcfg:localconfig):
        usuario: str = localcfg.get_db_user() # pyright: ignore[reportAssignmentType]
        senha: str = localcfg.get_db_password() # type: ignore
        host: str = localcfg.get_db_server() # type: ignore
        porta: int = localcfg.get_db_port() # type: ignore
        banco: str = localcfg.get_db_name()         # type: ignore
        

        self.engine = None
        self._SessionFactory = None        
        self._connect_database(usuario, senha, host, porta, banco)

    def _connect_database(self, usuario, senha, host, porta, banco):
        try:
            # boas opções para MySQL
            self.engine = create_engine(
                f"mysql+pymysql://{usuario}:{senha}@{host}:{porta}/{banco}",
                pool_pre_ping=True,      # evita 'MySQL server has gone away'
                future=True              # API 2.0 do SQLAlchemy
            )

            # testa conexão e fecha imediatamente
            with self.engine.connect() as conn:
                pass

            # guarda a fábrica de sessões
            self._SessionFactory = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)            
        except Exception as e:
            raise RuntimeError(f"[ERRO] Falha ao conectar com o banco de dados: {e}")


    def get_session(self):
        """Cria uma nova sessão (útil para contextos paralelos)."""
        if self._SessionFactory is None:
            raise RuntimeError("Banco não inicializado.")
        return self._SessionFactory()


    def dispose(self):
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None

    def test_connection(self):
        with self.engine.connect() as conn:
            return True
