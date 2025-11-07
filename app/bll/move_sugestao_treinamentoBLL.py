# move_sugestao_treinamentoBll.py
import string
from sys import exception
from typing import Optional
from sklearn.covariance import empirical_covariance
from sqlalchemy import text
from sqlalchemy.orm import Session
from torch import embedding
from common import print_with_time, print_error
import logger
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule
from bll.check_collidingBLL import check_collidingBLL as check_collidingBLLModule

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
            self.check_collidingBll = check_collidingBLLModule(session)
            self.ids_a_mover_qdrant_final = []
        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar move_sugestao_treinamentoBLL: {e}")

    #insere um idbase no qdrant na base final apaga da tabela de textos classificar e cadastra na tabela de textos treinamento
    def _move_ids_duplicados(self,list_duplicados,idBase: int, codclasse:int, classe:str):        
        try:                        
            registro  = self.qdrant_utils.get_id(id=idBase,collection_name=self.train_collection)
            if registro is None: #se não encontrar o idBase na collection de treinamento eu não posso mover
               raise RuntimeError(f"Erro em _get_ids_to_move id {idBase} não encontrado na collection {self.train_collection}")
            else:
                regbase = self.qdrant_utils.get_id(id=idBase,collection_name=self.final_collection)#verifica se existe na collection final
                self._move_to_textos_treinamento(idBase,codclasse)
                self.ids_a_mover_qdrant_final.append(idBase)#adiciona para mover depois que todos os duplicados forem apagados                
            
     
            for item in list_duplicados:
                self._delete_sugestao_textos_classificar(item)
            
        except Exception as e:
                raise RuntimeError(f"Erro inserindo Id  {idBase} _get_ids_to_move {e}")        

    #Verifica se o idBase já está na coleção final do Qdrant e, se não estiver, move-o para lá."""
    def check_idBase_in_final_collection(self, idBase: int, codclasse:int, classe:str)-> int:
        try:
            registro_final = self.qdrant_utils.get_id(id=idBase, collection_name=self.final_collection)
            if registro_final is None:
                registro_train = self.qdrant_utils.get_id(id=idBase, collection_name=self.train_collection)
                if registro_train is None:
                    raise RuntimeError(f"Aviso: ID {idBase} não encontrado na collection train, pulando")
                
                self._move_to_textos_treinamento(idBase, codclasse)  # Move para textos_treinamento                
                return 1#server para dizer que moveu e incrementar a contagem
            else:
                return 0
        except Exception as e:
            raise RuntimeError(f"Erro em check_idBase_in_final_collection para ID {idBase}: {e}")
    

    #Determina os IDs inferiores a min_similarity para mover para treinamento, move e apaga os ids duplicados (igual a 100) para a base de treinamento
    def _get_ids_to_move(self, idBase: int, idSimilar: int, codclasse:int, classe:str) -> tuple[list[int],int]:        
        qtdmovida=0
        ids_to_move = []
        lista_duplicados = []        
        query = f"""
                SELECT IdSimilar, Similaridade
                FROM sugestao_textos_classificar
                WHERE IdBase = :idBase and Similaridade >= {self.min_similarity}                
                order by IdSimilar
         """
        rows                = self.session.execute(text(query), {"idBase": idBase}).mappings().all()
        ids_to_move         = [row['IdSimilar'] for row in rows if row['Similaridade'] <  100]
        lista_duplicados    = [row['IdSimilar'] for row in rows if row['Similaridade'] >= 100]

        # Adiciona idBase se não estiver na collection final só deve inserir um para não ter duplicatas            
        if lista_duplicados:        
            self._move_ids_duplicados(lista_duplicados, idBase, codclasse, classe)          

        if (len(ids_to_move) == 0) and (len(lista_duplicados) == 0):#se não tiver nada é porque é porque não tem ninguem acima do min_similarity e ai deve pegar o idSimilar e o base
            ids_to_move.append(idSimilar)
            qtdmovida = self.check_idBase_in_final_collection(idBase, codclasse, classe)#Isso é necessario pois caso a semelhança não seja 100% o idBase pode não ter sido movido ainda
                                
        return list(set(ids_to_move)),(len(lista_duplicados)+qtdmovida)

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
            query_insert = f"""
                INSERT INTO textos_treinamento
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
        except Exception as e:
            raise RuntimeError(f"Erro ao mover para textos_treinamento: {e}")

    #Move ou atualiza um ponto da collection train para a collection final no Qdrant."""
    def _move_to_qdrant_final(self, id: int, embeddings: dict, CodClasse: int, classe:str) -> None:        
        #insere na colection final
        self.qdrant_utils.upinsert_id(collection_name=self.final_collection,
                                    id=id, 
                                    embeddings=embeddings,
                                    codclasse=CodClasse,
                                    classe=classe)
    
        #apaga da colection de treinamento
        self.qdrant_utils.delete_id(collection_name=self.train_collection, id=id)

    #Remove um ID registro da tabela textos_classificar
    def _delete_id_from_textos_classificar(self, id: int) -> None:        
        query_delete = "DELETE FROM textos_classificar WHERE id = :id"
        self.session.execute(text(query_delete), {"id": id})
        self.qdrant_utils.delete_id(collection_name=self.train_collection, id=id)        

    #Remove todos os registros relacionados ao idBase da tabela sugestao_textos_classificar.
    def _delete_sugestao_textos_classificar(self, idSimilar: int) -> None:                      
        try:
            query = f"DELETE FROM sugestao_textos_classificar WHERE IdSimilar = :IdSimilar"
            self.session.execute(text(query), {"IdSimilar": idSimilar})        
        except Exception as e:
            raise RuntimeError(f"Erro ao deletar id {idSimilar} sugestao_textos_classificar: {e}")
        
    def _move_ids_to_qdrant_final(self,CodClasse:int, Classe: str) -> None:
        try:
            for id in self.ids_a_mover_qdrant_final:
                registro = self.qdrant_utils.get_id(id=id, collection_name=self.train_collection)
                if registro is None:
                    raise RuntimeError(f"Aviso: ID {id} não encontrado na collection train, pulando")
                    continue
                
                self._move_to_qdrant_final(id, registro["Embedding"], CodClasse, Classe)
        except Exception as e:
            raise RuntimeError(f"Erro ao mover ids para Qdrant final: {e}") 
        
    def _check_reg_exists_in_sugestao_textos_classificar(self, idBase: int,idSimilar:int) -> bool:
        query = f"SELECT COUNT(*) FROM sugestao_textos_classificar WHERE IdBase = :IdBase AND IdSimilar = :IdSimilar"
        result = self.session.execute(text(query), {"IdBase": idBase, "IdSimilar": idSimilar}).scalar()
        return (result or 0) > 0

    def move_sugestao_treinamento(self, idBase: int, idSimilar: int, CodClasse: int) -> dict:
        try:    
            if not (self._check_reg_exists_in_sugestao_textos_classificar(idBase, idSimilar)):
                raise RuntimeError(f"Erro: Registro com IdBase {idBase} e IdSimilar {idSimilar} não encontrado em sugestao_textos_classificar")
            
            #Bloco que procura o idBase
            idFound = self.qdrant_utils.get_id(id=idBase, collection_name=self.train_collection)            
            if not idFound:
                idFound = self.qdrant_utils.get_id(id=idBase, collection_name=self.final_collection)#ele pode estar na coleção final pois ja foi movido anteriormente
                if not idFound:
                    raise RuntimeError(f"Erro: O IdBase {idBase} não foi encontrado na coleção de treinamento ou na coleção final.")                
            itens_colidentes = self.check_collidingBll.check_colliding_by_Embedding(idFound["Embedding"],idBase,CodClasse)
            if len(itens_colidentes) > 0:
                return {
                    "status": "ERROR",
                    "mensagem": f"""Foram encontradas colisões de classe para o idBase {idBase} fornecido.\n
                            Classe ja definida como {itens_colidentes[0]['Classe']}.\n
                            Impossível mover para treinamento.\n
                            Pressione 'Salvar Mesmo colidindo' caso queira forçar a classe.""",
                    "itens_colidentes": itens_colidentes
                }

            #Bloco que procura o idSimilar
            idFound = self.qdrant_utils.get_id(id=idSimilar, collection_name=self.train_collection)   
            if not idFound:
                idFound = self.qdrant_utils.get_id(id=idSimilar, collection_name=self.final_collection)#ele pode estar na coleção final pois ja foi movido anteriormente
                if not idFound:
                    raise RuntimeError(f"Erro: O IdSimilar {idSimilar} não foi encontrado na coleção de treinamento ou na coleção final.")                
            itens_colidentes = self.check_collidingBll.check_colliding_by_Embedding(idFound["Embedding"],idSimilar,CodClasse)
            if len(itens_colidentes) > 0:
                return {
                    "status": "ERROR",
                    "mensagem": f"""Foram encontradas colisões de classe para o idSimilar {idSimilar} fornecido.\n
                            Classe ja definida como {itens_colidentes[0]['Classe']}.\n 
                            Impossível mover para treinamento.\n 
                            Pressione 'Salvar Mesmo colidindo' caso queira forçar a classe.""",
                    "itens_colidentes": itens_colidentes
                }
            
            self.ids_a_mover_qdrant_final = []
            classe = self._get_classe(CodClasse)
            result = self._get_ids_to_move(idBase, idSimilar,CodClasse,classe)
            ids_to_move = result[0]# lista de ids a mover
            qtd_movida_igual = result[1] #quantidade de ids iguais 100 que já foram movidos para treinamento
            moved_ids = []

            #move os ids pro qdrant e apaga aqueles que tem similaridade > min_similarity e < 100 sendo que aquilo que é = 100 já foi movido anteriormente
            for id in ids_to_move:
                # Obtém o ponto da collection train
                id_Data = self.qdrant_utils.get_id(id=id, collection_name=self.train_collection)
                if (id_Data is None):
                    raise RuntimeError(f"Aviso: ID {id} não encontrado na collection train, pulando")
                    continue                
                        
                self.ids_a_mover_qdrant_final.append(id)#adiciona para mover depois que todos os duplicados forem apagados                              
                self._move_to_textos_treinamento(id, CodClasse)  # Move para textos_treinamento                
                moved_ids.append(id)

            #grava as mudanças no banco de dados idéia é que tudo seja feito numa transação só
            self.session.commit()
            #Agora move todos os ids para a coleção final do Qdrant
            self._move_ids_to_qdrant_final(CodClasse=CodClasse, Classe=classe)

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
