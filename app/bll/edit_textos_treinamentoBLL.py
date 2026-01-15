from ast import Dict, List
import os
from typing_extensions import runtime
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, Filter, FieldCondition, MatchValue
from sqlalchemy import RowMapping, Sequence, text
from tqdm import tqdm
from sqlalchemy.orm import Session
from common import print_with_time, print_error, get_localconfig
import bll.embeddingsBll as embeddingsBllModule
import gpu_utils as gpu_utilsModule
import logger
from collections.abc import Sequence
import torch
import time
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule
from bll.check_collidingBLL import check_collidingBLL as check_collidingBLLModule


class edit_textos_treinamentoBLL:
    def __init__(self, session: Session):
        from main import localconfig as localcfg        
        self._session = session
        self.qdrant_utils = Qdrant_UtilsModule()
        self.final_collection = self.qdrant_utils.get_collection_name("final")             
        self.qdrant_client = self.qdrant_utils.get_client()

    def _edit_texto_treinamento(self, id: int, codClasse: int, idFound: dict[str, any]):
        try:
            if not self.qdrant_utils.upinsert_id(collection_name=self.final_collection,
                                          id=id,
                                          embeddings=idFound["Embedding"],
                                          codclasse=codClasse,
                                          classe=idFound["Classe"]):
                raise RuntimeError(f"Erro ao atualizar o ID {id} na coleção final.")
                        
            sql = text("""
                UPDATE textos_treinamento
                SET cod_classe = :codClasse,DataHoraEdit = CURRENT_TIMESTAMP
                WHERE id = :id
            """)
            self._session.execute(sql, {"codClasse": codClasse, "id": id})
            self._session.commit()
            print_with_time(f"Texto de treinamento com ID {id} atualizado para a classe {codClasse}.")
        except Exception as e:
            self._session.rollback()
            raise RuntimeError(f"Erro ao editar o texto de treinamento com ID {id}: {str(e)}")            
              
    #edita o texto de treinamento em textos_treinamento
    def editar_texto_treinamento(self, id: int, codClasse: int):
         #Bloco que procura o id na coleção final
        idFound = self.qdrant_utils.get_id(id=id, collection_name=self.final_collection)            
        if not idFound:
            raise RuntimeError(f"Erro: O ID {id} não foi encontrado na coleção final.")

        check_collidingBLLModule_handle = check_collidingBLLModule(self._session)

        itens_colidentes = check_collidingBLLModule_handle.check_colliding_by_Embedding(idFound["Embedding"], id, codClasse)
        if len(itens_colidentes) > 0:
            return {
                "status": "ERROR",
                "mensagem": f"Foram encontradas colisões de classe para o ID {id} fornecido. Impossível mover para treinamento.",
                "itens_colidentes": itens_colidentes
                }


        self._edit_texto_treinamento(id, codClasse, idFound)
        
        return {
                    "status": "sucesso", 
                    "mensagem": f"Texto de treinamento com ID {id} atualizado para a classe {codClasse}."
                }