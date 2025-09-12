from sqlalchemy import Column, Integer, Numeric, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class IdsColidentes(Base):
    __tablename__ = "idscolidentes"
    __table_args__ = {'mysql_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_bin', 'mysql_engine': 'InnoDB'}

    Id = Column(Integer, nullable=False, primary_key=True)
    IdColidente = Column(Integer, nullable=False, primary_key=True)
    semelhanca = Column(Numeric(13, 4), nullable=True)    
