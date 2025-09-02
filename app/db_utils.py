# db_utils.py
from threading import local
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from common import print_error, print_with_time
import localconfig

class Db:
    def __init__(self,localcfg:localconfig):
        usuario: str = localcfg.get_db_user()
        senha: str = localcfg.get_db_password()
        host: str = localcfg.get_db_server()
        porta: int = localcfg.get_db_port()
        banco: str = localcfg.get_db_name()        
        print_with_time(f"Inicializando conexão com banco de dados {host}:{porta}/{banco}..."
                        )
        self.session = None
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
            self.session = self._SessionFactory()

            print_with_time("Conexão com banco de dados Ok")
        except Exception as e:
            raise RuntimeError(f"[ERRO] Falha ao conectar com o banco de dados: {e}")
            exit(1)

    def get_session(self):
        """Cria uma nova sessão (útil para contextos paralelos)."""
        if self._SessionFactory is None:
            raise RuntimeError("Banco não inicializado.")
        return self._SessionFactory()

    def close(self):
        """Fecha a sessão e o engine."""
        if self.session is not None:
            self.session.close()
            self.session = None
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None


