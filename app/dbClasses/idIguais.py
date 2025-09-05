from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.mysql import INTEGER as MySQLInteger

class Base(DeclarativeBase): pass

class IdsIguais(Base):
    __tablename__ = 'idsiguais'
    __table_args__ = {'mysql_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_bin', 'mysql_engine': 'InnoDB'}

    id: Mapped[int] = mapped_column(MySQLInteger(unsigned=True), primary_key=True, nullable=False)
    idIgual: Mapped[int] = mapped_column(MySQLInteger(unsigned=True), primary_key=True, nullable=False)
