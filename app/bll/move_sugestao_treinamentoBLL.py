# move_sugestao_treinamentoBll.py
from sys import exception
from typing import Optional
from sklearn.covariance import empirical_covariance
from sqlalchemy import text
from sqlalchemy.orm import Session
from torch import embedding
from common import print_with_time, print_error
import logger
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule

class move_sugestao_treinamentoBLL:
    def __init__(self, session: Session):
        """
        Inicializa a classe para mover sugestões de classificação para treinamento usando Qdrant e SQL.
        Args:
            session (Session): Sessão SQLAlchemy para operações no banco.
        """
        try:
            from main import localconfig as localcfg
            self.session = session
            self.localconfig = localcfg
            self.qdrant_utils = Qdrant_UtilsModule()
            self.qdrant_client = self.qdrant_utils.get_client()
            self.train_collection = self.qdrant_utils.get_collection_name("train")
            self.final_collection = self.qdrant_utils.get_collection_name("final")
            self.qdrant_utils.create_collection(self.train_collection)
            self.qdrant_utils.create_collection(self.final_collection)
            self.logger = logger.log
            self.min_similarity =  98.5
        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar move_sugestao_treinamentoBLL: {e}")

    #insere um idbase no qdrant na base final apaga da tabela de textos classificar e cadastra na tabela de textos treinamento
    def _move_ids_duplicados(self,list_duplicados,idBase: int, codclasse:int, classe:str):        
        try:
            if not self.qdrant_utils.get_id(id=idBase, collection_name=self.final_collection):
                registro  = self.qdrant_utils.get_id(id=idBase,collection_name=self.train_collection)
                if registro is None:
                   raise RuntimeError(f"Erro em _get_ids_to_move id {idBase} não encontrado na base para classificar")
                                                   
                self._move_to_qdrant_final(id,registro["Embedding"],codclasse,classe)        
                self._move_to_textos_treinamento(idBase,codclasse)

        except Exception as e:
             raise RuntimeError(f"Erro inserindo Id  {idBase} _get_ids_to_move")
     
        for item in list_duplicados:
            self._delete_from_textos_classificar(item)



    def _get_ids_to_move(self, idBase: int, idSimilar: int, tipo_id: str,codclasse:int, classe:str) -> list[int]:
        """Determina os IDs a serem movidos com base no tipo_id."""
        ids_to_move = []
        if tipo_id == "idsimilar":
            query = f"""
                SELECT IdSimilar, Similaridade
                FROM sugestao_textos_classificar
                WHERE IdBase = :idBase AND Similaridade >= {self.min_similarity}
                order by IdSimilar
            """
            rows         = self.session.execute(text(query), {"idBase": idBase}).mappings().all()
            ids_to_move  = [row['IdSimilar'] for row in rows if row['Similaridade'] < 100]
            lista_duplicados     = [row['IdSimilar'] for row in rows if row['Similaridade'] >= 100]
            
            # Adiciona idBase se não estiver na collection final só deve inserir um para não ter duplicatas            
            if lista_duplicados:        
                    self._move_ids_duplicados(lista_duplicados, idBase, codclasse, classe)          

                    
        return list(set(ids_to_move))

    def _get_classe(self,codclasse) -> str:    
        try:
            query = f"SELECT Classe from classes where codclasse = :codclasse"

            result = self.session.execute(text(query),{"codclasse":codclasse}).mappings().all()
            if not result :
                raise RuntimeError(f"Erro em _get_classe CodClasse {codclasse} não encontrada")                
            
            return result[0]["Classe"]
        
        except Exception as e:
            raise RuntimeError(f"Erro ao obter classe em _get_classe: {e}")
        

    def _get_text_data(self, id_: int) -> Optional[dict]:
        """Obtém dados do texto da tabela textos_classificar."""
        query_get = """
            SELECT DataEvento, Documento, UF, TxtDocumento, TxtTreinamento, QtdPalavras,
                   TipoDefinicaoInicioTxt, ProcessadoNulo, PalavraIni
            FROM textos_classificar
            WHERE id = :id
        """
        row = self.session.execute(text(query_get), {"id": id_}).mappings().first()
        if row is None:
            print_with_time(f"Aviso: ID {id_} não encontrado em textos_classificar, pulando")
            return None
        return dict(row)
    
    #Move um registro para a tabela textos_treinamento
    def _move_to_textos_treinamento(self, id: int, CodClasse: int) -> None:
        query_insert = """
            INSERT INTO ignore textos_treinamento
            (id, DataEvento, Documento, CodClasse, UF, TxtDocumento, TxtTreinamento, QtdPalavras,
            TipoDefinicaoInicioTxt, ProcessadoNulo, PalavraIni, Indexado, BuscouIgual, BuscouColidente)
            SELECT id, DataEvento, Documento, :CodClasse, UF, TxtDocumento, TxtTreinamento, QtdPalavras,
                TipoDefinicaoInicioTxt, ProcessadoNulo, PalavraIni, 1, 0, 0
            FROM textos_classificar
            WHERE id = :id
            ON DUPLICATE KEY UPDATE
                CodClasse = VALUES(CodClasse),
                TxtTreinamento = VALUES(TxtTreinamento),
                TxtDocumento = VALUES(TxtDocumento)
        """
        self.session.execute(text(query_insert), {"id": id, "CodClasse": CodClasse})

        self._delete_sugestao_textos_classificar(id)        


    #Remove um registro da tabela textos_classificar
    def _delete_from_textos_classificar(self, id: int) -> None:        
        query_delete = """
            DELETE FROM textos_classificar WHERE id = :id
        """
        self.session.execute(text(query_delete), {"id": id})

    #Move ou atualiza um ponto da collection train para a collection final no Qdrant."""
    def _move_to_qdrant_final(self, id: int, embeddings: dict, CodClasse: int, classe:str) -> None:        
        #insere na colection final
        self.qdrant_utils.upinsert_id(collection_name=self.final_collection,
                                    id=id, 
                                    embeddings=embeddings,
                                    codclasse=CodClasse,
                                    classe=classe)
    
        #apaga da colection de classificação
        self.qdrant_utils.delete_id(collection_name=self.train_collection, id=id)

    #Remove todos os registros relacionados ao idBase da tabela sugestao_textos_classificar.
    def _delete_sugestao_textos_classificar(self, id: int) -> None:        
        query_delete_sug = "DELETE FROM sugestao_textos_classificar WHERE IdSimilar = :id"
        self.session.execute(text(query_delete_sug), {"id": id})
        
    def move_to_treinamento(self, idBase: int, idSimilar: int, CodClasse: int, tipo_id: str) -> dict:
        try:
            if tipo_id not in ["idbase", "idsimilar"]:
                raise RuntimeError("tipo_id deve ser 'idbase' ou 'idsimilar'")            
    
            classe = self._get_classe(CodClasse)    
            ids_to_move = self._get_ids_to_move(idBase, idSimilar, tipo_id,CodClasse,classe)
            moved_ids = []

            #move os ids pro qdrant e apaga aqueles que tem similaridade > min_similarity e < 100
            for id in ids_to_move:
                # Obtém o ponto da collection train
                embeddings = self.qdrant_utils.get_id(id=id, collection_name=self.train_collection)
                if embeddings is None:
                    print_with_time(f"Aviso: ID {id} não encontrado na collection train, pulando")
                    continue                
        
                # Move para Qdrant final (insere ou atualiza)
                self._move_to_qdrant_final(id, embeddings["Embedding"], CodClasse, classe)

                # Move para textos_treinamento
                self._move_to_textos_treinamento(id, CodClasse)

                # Deleta de textos_classificar
                self._delete_from_textos_classificar(id)
                moved_ids.append(id)


            self.session.commit()

            sucessMessage = f"Movidos {len(moved_ids)} registros para treinamento e Qdrant final"
            print_with_time(sucessMessage)
            return {
                "status": "OK",
                "mensagem": sucessMessage,
                "movidos": moved_ids
            }

        except Exception as e:
            self.session.rollback()
            errorMessage = f"Erro ao mover sugestões para treinamento: {e}"
            print_with_time(errorMessage)
            return {
                "status": "ERROR",
                "mensagem": errorMessage
            }
