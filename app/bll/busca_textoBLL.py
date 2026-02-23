#busca_textoBLL.py
import re
import string
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from requests import session
from sqlalchemy import text
from bll.embeddingsBll import EmbeddingsBll  # Importing the original module
from collections import defaultdict
from transformers import AutoTokenizer
from sqlalchemy.orm import Session
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule
from bll.classifica_textoBll import classifica_textoBll as classifica_textoBllModule
from bll.classifica_textoBll import classifica_textoBll
import qdrant_utils

class busca_textoBLL:
    def __init__(self, embeddingsModule: EmbeddingsBll, session: Session):             
        self.embeddingsModule = embeddingsModule                  
        self.session = session  
        self.qdrant_utils = Qdrant_UtilsModule()  # Initialize Qdrant_Utils instance
        self.collections = self.qdrant_utils.list_collections()     
            

    ### Busca texto e retorna similaridades  
    def busca_texto(self, 
                    id:int,
                    codclasse:int,
                    texto:str,
                    similaridade_minima:float,
                    collection_name:str
                    ) -> list[dict]:        
        try:
           if (len(texto.strip()) < 3):
               raise RuntimeError("O texto para busca deve ter mais de 3 caracteres.")

           collection_name = self.qdrant_utils.get_collection_name(collection_name)
           if collection_name not in self.collections:
               raise RuntimeError(f"A collection '{collection_name}' não existe no banco vetorial.")

           # Generate embedding for the input text to compare in future
           query_embedding = self.embeddingsModule.generate_embedding(text = texto, Id = 0) # Id = 0 é porque não importa nesse caso
                
           return self.qdrant_utils.search_embedding_and_metaData(embedding = query_embedding,
                                                         collection_name = collection_name,
                                                         itens_limit = 100,
                                                         similarity_threshold = similaridade_minima,
                                                         id = id,
                                                         codclasse = codclasse)
                            
        except Exception as e:
            raise RuntimeError(f"Erro em busca_texto: {e}")