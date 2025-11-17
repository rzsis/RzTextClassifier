# move_sugestao_treinamentoBll.py
from ast import Dict
from datetime import datetime
import string
from sys import exception
from typing import Any, Optional
from networkx import reconstruct_path
from pkg_resources import UnknownExtra
from sklearn.covariance import empirical_covariance
from sqlalchemy import text
from sqlalchemy.orm import Session
from torch import embedding
from common import print_with_time, print_error
import logger
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule
from bll.check_collidingBLL import check_collidingBLL as check_collidingBLLModule
from bll.salva_log_AlteracoesBll import salva_log_AlteracoesBll as salva_log_AlteracoesBllModule
import bll.embeddingsBll as embeddingsBllModule

class move_sugestao_treinamentoBLL:
    
    #Inicializa a classe para mover sugestões de classificação do banco de dados para treinamento para o oficial usando Qdrant e SQL
    def __init__(self, session: Session):
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
            self.salva_log_alteracoesBll = salva_log_AlteracoesBllModule(session)            
            embeddingsBllModule.initBllEmbeddings(self.session)
            self.embeddingsBll = embeddingsBllModule.bllEmbeddings

        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar move_sugestao_treinamentoBLL: {e}")

    def _recreate_IdBase(self, idBase:int) -> Optional[dict[str, Any]]:
        try:
            row = self.session.execute(text("""
                SELECT TxtTreinamento
                FROM textos_classificar
                WHERE id = :idBase
            """), {"idBase": idBase}).mappings().first()
            if not row:
                raise RuntimeError(f"Erro em _recreate_IdBase O ID {idBase} não foi encontrado na tabela textos_classificar.")
            
            texto_treinamento = row["TxtTreinamento"]
            embedding_vector = self.embeddingsBll.generate_embedding(texto_treinamento,Id=idBase)

            if embedding_vector is None:
                raise RuntimeError(f"Erro ao gerar embedding para o ID {idBase}.")
            
            self.qdrant_utils.upinsert_id(
                collection_name=self.train_collection,
                id=idBase,
                embeddings=embedding_vector,
                codclasse=None,
                classe=None
            )
            print_with_time(f"Embedding recriado e inserido na collection final para o ID {idBase} na coleção de treinamento.")

            registro = self.qdrant_utils.get_id(id=idBase, collection_name=self.train_collection)
            if not registro:
                raise RuntimeError(f"Erro: O ID {idBase} não foi encontrado na coleção de treinamento após a inserção, mesmo após a recriação do embedding.")
            
            return registro
        
        except Exception as e:
            raise RuntimeError(f"Erro ao recriar embedding para o ID {idBase}: {e}")
        

    #insere um idbase no qdrant na base final apaga da tabela de textos classificar e cadastra na tabela de textos treinamento
    def _move_ids_duplicados(self,list_duplicados,idBase: int, codclasse:int, coduser:int) -> None:        
        try:                        
            registro  = self.qdrant_utils.get_id(id=idBase,collection_name=self.train_collection)
            if registro is None: #se não encontrar o idBase na collection de treinamento eu gero um novo embedding em insiro ele denovo
                registro = self._recreate_IdBase(idBase)
            
            
            self._move_to_textos_treinamento(idBase,codclasse,coduser)  # Move para textos_treinamento
            self.ids_a_mover_qdrant_final.append(idBase)#adiciona para mover depois que todos os duplicados forem apagados                
            
     
            for item in list_duplicados:
                self._delete_sugestao_textos_classificar(item)
            
        except Exception as e:
                raise RuntimeError(f"Erro inserindo Id  {idBase} _get_ids_to_move {e}")        

    #Verifica se o idBase já está na coleção final do Qdrant e, se não estiver, move-o para lá."""
    def check_idBase_in_final_collection(self, idBase: int, codclasse:int, coduser:int)-> int:
        try:
            registro_final = self.qdrant_utils.get_id(id=idBase, collection_name=self.final_collection)
            if registro_final is None:
                registro_train = self.qdrant_utils.get_id(id=idBase, collection_name=self.train_collection)
                if registro_train is None:
                    raise RuntimeError(f"Aviso: ID {idBase} não encontrado na collection train, pulando")
                
                self._move_to_textos_treinamento(idBase, codclasse, coduser)  # Move para textos_treinamento                
                return 1#server para dizer que moveu e incrementar a contagem
            else:
                return 0
        except Exception as e:
            raise RuntimeError(f"Erro em check_idBase_in_final_collection para ID {idBase}: {e}")

    #Função desta rotina é quanto um texto que não tem 100% de similaridade mais tem mesmo textos nos similares retorne todos os similares para migrar mais textos por vez.
    def _get_ids_similares_adicionais(self, idBase: int, idSimilar: int, ids_similares: list[int]) -> list[int]:      
        try:
            query = f"""
                SELECT tc.TxtTreinamento
                    from  textos_classificar tc
                    WHERE  tc.id in (select stc.idSimilar from sugestao_textos_classificar stc
	                    where IdBase = {idBase} and idSimilar = {idSimilar})
            """
    
            row = self.session.execute(text(query)).mappings().all()
            if not row:
                return ids_similares
            
            texto_treinamento = row[0]['TxtTreinamento']
            query = f"""
                    SELECT tc.id, tc.TxtTreinamento
                        from  textos_classificar tc
                        inner join sugestao_textos_classificar stc on tc.id = stc.IdSimilar 
                        WHERE  tc.TxtTreinamento = '{texto_treinamento}'
                        and tc.id in (select IdSimilar  from sugestao_textos_classificar where idBase = {idBase})
            """

            rows = self.session.execute(text(query)).mappings().all()    
            if not rows:
                return ids_similares
                                              
            for row in rows:
                if (row not in ids_similares):
                    ids_similares.append(row)

            return ids_similares    
        
        except Exception as e:
            raise RuntimeError(f"Erro ao obter IDs iguais adicionais para IDBase {idBase}: {e}")

    #Determina os IDs inferiores a min_similarity para mover para treinamento, move e apaga os ids duplicados (igual a 100) para a base de treinamento
    def _get_ids_to_move(self, idBase: int, idSimilar: int, codclasse:int, classe:str, coduser:int) -> tuple[list[int],int]:        
        qtdmovida=0
        lista_similares = []
        lista_iguais = []        
        query = f"""
                SELECT IdSimilar, Similaridade
                FROM sugestao_textos_classificar
                WHERE IdBase = :idBase and Similaridade >= {self.min_similarity}                
                order by IdSimilar
         """
        rows                = self.session.execute(text(query), {"idBase": idBase}).mappings().all()
        lista_iguais        = [row['IdSimilar'] for row in rows if row['Similaridade'] >= 100]#lista com os ids iguais a 100 de similaridade        
        lista_similares     = [row['IdSimilar'] for row in rows if row['Similaridade'] <  100] #lista com os ids acima do min_similarity e abaixo de 100


        # Adiciona idBase se não estiver na collection final só deve inserir um para não ter duplicatas            
        if lista_iguais:           
            self._move_ids_duplicados(lista_iguais, idBase, codclasse, coduser)          

        if (len(lista_similares) == 0) and (len(lista_iguais) == 0):#se não tiver nada é porque não tem ninguem acima do min_similarity e ai deve pegar o idSimilar e o base
            lista_similares.insert(0, idBase)
            qtdmovida = self.check_idBase_in_final_collection(idBase, codclasse, coduser)#Isso é necessario pois caso a semelhança não seja 100% o idBase pode não ter sido movido ainda

        lista_similares = self._get_ids_similares_adicionais(idBase=idBase, idSimilar=idSimilar, ids_similares=lista_similares)
                                                    
        return list(set(lista_similares)),(len(lista_iguais)+qtdmovida)

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
    
    #Move um registro para a tabela textos_treinamento
    def _move_to_textos_treinamento(self, id: int, CodClasse: int, CodUser: int) -> None:
        try: 
            agora = datetime.now()
            query_insert = f"""
                INSERT INTO textos_treinamento
                (id, DataEvento, Documento, CodClasse, UF, TxtDocumento, TxtTreinamento, QtdPalavras,
                TipoDefinicaoInicioTxt, ProcessadoNulo, PalavraIni, Indexado, BuscouIgual, BuscouColidente,DataHoraInsert,DataHoraEdit)
                SELECT id, DataEvento, Documento, :CodClasse, UF, TxtDocumento, TxtTreinamento, QtdPalavras,
                    TipoDefinicaoInicioTxt, ProcessadoNulo, PalavraIni, 1, 0, 0, :DataHoraInsert, :DataHoraEdit
                FROM textos_classificar
                WHERE id = :id
                ON DUPLICATE KEY UPDATE
                    CodClasse = VALUES(CodClasse),
                    TxtTreinamento = VALUES(TxtTreinamento),
                    TxtDocumento = VALUES(TxtDocumento)
            """
            self.session.execute(text(query_insert), {"id": id, "CodClasse": CodClasse, "DataHoraInsert": agora, "DataHoraEdit": agora})
            self.salva_log_alteracoesBll.insert_log_texto_treinamento([id], CodUser, auto_commit=False)
            

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

    #Remove todos os registros relacionados ao idBase da tabela sugestao_textos_classificar.
    def _delete_sugestao_textos_classificar(self, idSimilar: int) -> None:                      
        try:
            query = f"DELETE FROM sugestao_textos_classificar WHERE IdSimilar = :IdSimilar"
            self.session.execute(text(query), {"IdSimilar": idSimilar})        
        except Exception as e:
            raise RuntimeError(f"Erro ao deletar id {idSimilar} sugestao_textos_classificar: {e}")
        
    #apaga textos duplicados na tabela textos_treinamento que foram movidos nesta batch
    def _delete_textos_duplicados_treinamento(self):
        qtdDuplicadosMovidos = 0    
        ids_str = ",".join(str(i) for i in self.ids_a_mover_qdrant_final)        
        sql = f"""
            select t.id,t.TxtTreinamento, Count(*) As QtdDuplicado, GROUP_CONCAT(t.id ) as IdsDuplicados from textos_treinamento t
                where t.id in ({ids_str})
                group by t.TxtTreinamento
                HAVING Count(*) > 1 
            """
        rows = self.session.execute(text(sql)).mappings().all()
        for row in rows:
            for id_to_delete in row["IdsDuplicados"].split(",")[1:]: #mantém o primeiro, deleta os outros
                delete_sql = "DELETE FROM textos_treinamento WHERE id = :id"
                self.session.execute(text(delete_sql), {"id": int(id_to_delete)})
                self.qdrant_utils.delete_id(collection_name=self.final_collection, id=int(id_to_delete))
                qtdDuplicadosMovidos += 1
        
        print_with_time(f"Removidos {qtdDuplicadosMovidos} textos duplicados na tabela textos_treinamento e Qdrant final.")
        self.session.commit()
                        
    #Move os IDs coletados para a coleção final do Qdrant.                    
    def _move_ids_to_qdrant_final(self,CodClasse:int, Classe: str) -> None:
        try:
            for id in self.ids_a_mover_qdrant_final:
                registro = self.qdrant_utils.get_id(id=id, collection_name=self.train_collection)
                if registro is None:
                    raise RuntimeError(f"Aviso: ID {id} não encontrado na collection train, pulando")
                    continue
                
                self._move_to_qdrant_final(id, registro["Embedding"], CodClasse, Classe)

            self._delete_textos_duplicados_treinamento()#apaga os textos duplicados na tabela textos_treinamento e no qdrant final
                
        except Exception as e:
            raise RuntimeError(f"Erro ao mover ids para Qdrant final: {e}") 
        
    def _check_reg_exists_in_sugestao_textos_classificar(self, idBase: int,idSimilar:int) -> bool:
        query = f"SELECT COUNT(*) FROM sugestao_textos_classificar WHERE IdBase = :IdBase AND IdSimilar = :IdSimilar"
        result = self.session.execute(text(query), {"IdBase": idBase, "IdSimilar": idSimilar}).scalar()
        return (result or 0) > 0

    #verifica se o idBase colidem com outras classes e retorna um erro caso colida para parar a movimentação
    def _check_coliding_idBase(self, idBase:int, codClasse:int, mover_com_colidencia:bool) -> dict | None:
        try:
            if mover_com_colidencia:               
                return None

            idFound = self.qdrant_utils.get_id(id=idBase, collection_name=self.train_collection)            
            if not idFound:
                idFound = self.qdrant_utils.get_id(id=idBase, collection_name=self.final_collection)#ele pode estar na coleção final pois ja foi movido anteriormente
                if not idFound:
                    raise RuntimeError(f"Erro: O IdBase {idBase} não foi encontrado na coleção de treinamento ou na coleção final.")                

            itens_colidentes = self.check_collidingBll.check_colliding_by_Embedding(idFound["Embedding"],idBase,codClasse)
            if len(itens_colidentes) > 0:
                    return {
                        "status": "ERROR",
                        "mensagem": f"""Foram encontradas colisões de classe para o idBase {idBase} fornecido.\n
                                Classe ja definida como {itens_colidentes[0]['Classe']}.\n
                                Impossível mover para treinamento.\n
                                Pressione 'Salvar Mesmo colidindo' caso queira forçar a classe.""",
                        "itens_colidentes": itens_colidentes
                    }
                
            return None
        
        except Exception as e:
            raise RuntimeError(f"Erro ao verificar colisão para _check_coliding_idBase {idBase}: {e}")

    #verifica se o idSimilar colidem com outras classes e retorna um erro caso colida para parar a movimentação
    def _check_coliding_idSimilar(self, idSimilar:int, codClasse:int, mover_com_colidencia:bool) -> dict | None:
        try:
            if mover_com_colidencia:               
                return None
                        
            idFound = self.qdrant_utils.get_id(id=idSimilar, collection_name=self.train_collection)   
            if not idFound:
                idFound = self.qdrant_utils.get_id(id=idSimilar, collection_name=self.final_collection)#ele pode estar na coleção final pois ja foi movido anteriormente
                if not idFound:
                    raise RuntimeError(f"Erro: O IdSimilar {idSimilar} não foi encontrado na coleção de treinamento ou na coleção final.")                
                            
            itens_colidentes = self.check_collidingBll.check_colliding_by_Embedding(idFound["Embedding"],idSimilar,codClasse)
            if len(itens_colidentes) > 0:
                    return {
                        "status": "ERROR",
                        "mensagem": f"""Foram encontradas colisões de classe para o idSimilar {idSimilar} fornecido.\n
                                Classe ja definida como {itens_colidentes[0]['Classe']}.\n 
                                Impossível mover para treinamento.\n 
                                Pressione 'Salvar Mesmo colidindo' caso queira forçar a classe.""",
                        "itens_colidentes": itens_colidentes
                    }
                
            return None        
        except Exception as e:  
            raise RuntimeError(f"Erro ao verificar colisão para _check_coliding_idSimilar {idSimilar}: {e}")    
                
        
    #Main method to move suggested training texts based on similarity and class.
    #CodUser vem da interface do usuario
    #mover_com_colidencia serve para ignorar a verificação de colisão de classes caso o usuario queira forçar a movimentação
    def move_sugestao_treinamento(self, idBase: int, idSimilar: int, codClasse, coduser:int , mover_com_colidencia:bool=False) -> dict:
        try:    
            if not (self._check_reg_exists_in_sugestao_textos_classificar(idBase, idSimilar)):
                raise RuntimeError(f"Erro: Registro com IdBase {idBase} e IdSimilar {idSimilar} não encontrado em sugestao_textos_classificar")
            
            #Bloco que procura o colidencias com idBase
            result = self._check_coliding_idBase(idBase=idBase,codClasse=codClasse,mover_com_colidencia=mover_com_colidencia)
            if (result is not None):
                return result
            
            #Bloco que procura o colidencias com idSimilar
            result = self._check_coliding_idSimilar(idSimilar=idSimilar,codClasse=codClasse,mover_com_colidencia=mover_com_colidencia)
            if (result is not None):
                return result
       
            self.ids_a_mover_qdrant_final = []
            classe = self._get_classe(codClasse)
            result = self._get_ids_to_move(idBase, idSimilar, codClasse, classe, coduser)
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
                self._move_to_textos_treinamento(id, codClasse, coduser)  # Move para textos_treinamento                
                moved_ids.append(id)
                  
            
            #grava as mudanças no banco de dados idéia é que tudo seja feito numa transação só
            self.session.commit()
            #Agora move todos os ids para a coleção final do Qdrant
            self._move_ids_to_qdrant_final(CodClasse=codClasse, Classe=classe)

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
        finally:
            self.session.close()
