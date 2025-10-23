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
            #se ele não encontrar o idbase na collection final insere
            if not self.qdrant_utils.get_id(id=idBase, collection_name=self.final_collection):
                registro  = self.qdrant_utils.get_id(id=idBase,collection_name=self.train_collection)
                if registro is None:
                   raise RuntimeError(f"Erro em _get_ids_to_move id {idBase} não encontrado na collection {self.train_collection}")

                self._move_to_textos_treinamento(idBase,codclasse)                                                   
                self._move_to_qdrant_final(idBase,registro["Embedding"],codclasse,classe)        
                self.session.commit()
        except Exception as e:
             raise RuntimeError(f"Erro inserindo Id  {idBase} _get_ids_to_move {e}")
     
        for item in list_duplicados:
            self._delete_from_textos_classificar(item)

        self.session.commit()


    #Determina os IDs inferiores a min_similarity para mover para treinamento, move e apaga os ids duplicados (igual a 100) para a base de treinamento
    def _get_ids_to_move(self, idBase: int, idSimilar: int, codclasse:int, classe:str) -> tuple[list[int],int]:        
        ids_to_move = []
        lista_duplicados = []        
        query = f"""
                SELECT IdSimilar, Similaridade
                FROM sugestao_textos_classificar
                WHERE IdBase = :idBase AND Similaridade >= {self.min_similarity}
                order by IdSimilar
         """
        rows                = self.session.execute(text(query), {"idBase": idBase}).mappings().all()
        ids_to_move         = [row['IdSimilar'] for row in rows if row['Similaridade'] < 100]
        lista_duplicados    = [row['IdSimilar'] for row in rows if row['Similaridade'] >= 100]

        # Adiciona idBase se não estiver na collection final só deve inserir um para não ter duplicatas            
        if lista_duplicados:        
            self._move_ids_duplicados(lista_duplicados, idBase, codclasse, classe)          
                    
        return list(set(ids_to_move)),len(lista_duplicados)

    #obtem a classe pelo codclasse
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
        try: 
            query_insert = """
                INSERT ignore INTO textos_treinamento
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
            self.session.commit()
            self._delete_id_sugestao_textos_classificar(id)                    
        except Exception as e:
            raise RuntimeError(f"Erro ao mover para textos_treinamento: {e}")

        
    #Remove um registro da tabela textos_classificar
    def _delete_from_textos_classificar(self, id: int) -> None:        
        query_delete = " DELETE FROM textos_classificar WHERE id = :id"
        self.session.execute(text(query_delete), {"id": id})
        self.qdrant_utils.delete_id(collection_name=self.train_collection, id=id)
        self._delete_id_sugestao_textos_classificar(id)   

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
    def _delete_id_sugestao_textos_classificar(self, idSimilar: int) -> None:        
        try:
            query = "DELETE FROM sugestao_textos_classificar WHERE IdSimilar = :IdSimilar"
            self.session.execute(text(query), {"IdSimilar": idSimilar})        
        except Exception as e:
            raise RuntimeError(f"Erro ao deletar id {idSimilar} sugestao_textos_classificar: {e}")
        
    def move_to_treinamento(self, idBase: int, idSimilar: int, CodClasse: int, tipo_id: str) -> dict:
        try:
            if tipo_id not in ["idbase", "idsimilar"]:
                raise RuntimeError("tipo_id deve ser 'idbase' ou 'idsimilar'")            
    
            classe = self._get_classe(CodClasse)    
            result = self._get_ids_to_move(idBase, idSimilar,CodClasse,classe)
            ids_to_move = result[0]# lista de ids a mover
            qtd_movida_igual = result[1] #quantidade de ids iguais 100 que já foram movidos para treinamento
            moved_ids = []

            #move os ids pro qdrant e apaga aqueles que tem similaridade > min_similarity e < 100 sendo que aquilo que é = 100 já foi movido anteriormente
            for id in ids_to_move:
                # Obtém o ponto da collection train
                id_Data = self.qdrant_utils.get_id(id=id, collection_name=self.train_collection)
                if id_Data is None:
                    print_with_time(f"Aviso: ID {id} não encontrado na collection train, pulando")
                    continue                
        
                # Move para Qdrant final (insere ou atualiza)
                self._move_to_qdrant_final(id, id_Data["Embedding"], CodClasse, classe)

                # Move para textos_treinamento
                self._move_to_textos_treinamento(id, CodClasse)

                # Deleta de textos_classificar
                self._delete_from_textos_classificar(id)
                moved_ids.append(id)


            self.session.commit()

            total_movido = len(moved_ids) + qtd_movida_igual
            sucessMessage = f"Movidos {total_movido} registros para treinamento e Qdrant final"
            print_with_time(sucessMessage)
            return {
                "status": "OK",
                "mensagem": sucessMessage,
                "movidos": total_movido
            }

        except Exception as e:
            self.session.rollback()
            errorMessage = f"Erro ao mover sugestões para treinamento: {e}"
            print_with_time(errorMessage)
            return {
                "status": "ERROR",
                "mensagem": errorMessage
            }
