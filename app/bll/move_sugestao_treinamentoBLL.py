# move_sugestao_treinamentoBll.py
from ast import Dict
from datetime import datetime
import string
from sys import exception
from typing import Any, Optional
from networkx import reconstruct_path
from pkg_resources import UnknownExtra
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session
from sympy import Id
from torch import embedding
from common import print_with_time, print_error
import logger
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule
from bll.check_collidingBLL import check_collidingBLL as check_collidingBLLModule
from bll.salva_log_AlteracoesBll import salva_log_AlteracoesBll as salva_log_AlteracoesBllModule
import bll.embeddingsBll as embeddingsBllModule
import numpy as np

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
            self.salva_log_alteracoesBll = salva_log_AlteracoesBllModule(session)            
            embeddingsBllModule.initBllEmbeddings(self.session)
            self.embeddingsBll = embeddingsBllModule.bllEmbeddings

        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar move_sugestao_treinamentoBLL: {e}")
        
    #Recria o embedding para um idBase que não foi encontrado na coleção de treinamento do Qdrant.
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
                cod_classe=None                
            )
            print_with_time(f"Embedding recriado e inserido na collection final para o ID {idBase} na coleção de treinamento.")

            registro = self.qdrant_utils.get_id(id=idBase, collection_name=self.train_collection)
            if not registro:
                raise RuntimeError(f"Erro: O ID {idBase} não foi encontrado na coleção de treinamento após a inserção, mesmo após a recriação do embedding.")
            
            return registro
        
        except Exception as e:
            raise RuntimeError(f"Erro ao recriar embedding para o ID {idBase}: {e}")
         

    #Função desta rotina é quanto um texto que não tem 100% de similaridade mais tem mesmo textos nos similares 
    #retorne todos os similares para migrar mais textos por vez.
    def _get_ids_similares_adicionais(self, idBase: int, idSimilar: int, lista_similares: list[int], lista_iguais: list[int]) -> list[int]:      
        try:
            query = f"""
                SELECT tc.TxtTreinamento
                    from  textos_classificar tc
                    WHERE  tc.id in (select stc.idSimilar from sugestao_textos_classificar stc
	                    where IdBase = {idBase} and idSimilar = {idSimilar})
            """
    
            row = self.session.execute(text(query)).mappings().all()
            if not row:
                return lista_similares
            
            texto_treinamento = row[0]['TxtTreinamento']
            query = f"""
                    SELECT tc.id
                        from  textos_classificar tc
                        inner join sugestao_textos_classificar stc on tc.id = stc.IdSimilar 
                        WHERE  tc.TxtTreinamento = '{texto_treinamento}'
                        and tc.id in (select IdSimilar  from sugestao_textos_classificar where idBase = {idBase})
            """

            rows = self.session.execute(text(query)).scalars().all()    
            if not rows:
                return lista_similares
                                              
            for row in rows:
                if (row not in lista_similares) and (row not in lista_iguais):
                    lista_similares.append(row)

            return lista_similares    
        
        except Exception as e:
            raise RuntimeError(f"Erro ao obter IDs iguais adicionais para IDBase {idBase}: {e}")

    #Determina os IDs inferiores a min_similarity para mover para treinamento, move e apaga os ids duplicados (igual a 100) para a base de treinamento
    # 0 Retorna a Lista de similares
    # 1 Retorna a quantidade de ids iguais movidos
    # 2 Retorna a lista de ids iguais movidos
    def _get_ids_to_move(self, idBase: int, idSimilar: int, codclasse:int, classe:str, coduser:int) -> tuple[set[int],set[int]]:        
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
        

        if (len(lista_similares) == 0) and (len(lista_iguais) == 0):#se não tiver nada é porque não tem ninguem acima do min_similarity e ai deve pegar o idSimilar e o base
            lista_similares.insert(0, idBase)            

        lista_similares = self._get_ids_similares_adicionais(idBase=idBase, idSimilar=idSimilar, 
                                                             lista_similares=lista_similares, 
                                                             lista_iguais=lista_iguais)
                                                    
        return set(lista_similares), set(lista_iguais)

    #obtem a classe pelo codclasse
    def _get_classe(self,codclasse) -> str:    
        try:
            query = f"SELECT Classe from classes where codclasse = :codclasse"

            result = self.session.execute(text(query),{"codclasse":codclasse}).mappings().first()
            if not result :
                raise ValueError(f"Erro em _get_classe CodClasse {codclasse} não encontrada")                
            
            return result["Classe"]
        
        except ValueError:
            raise  # mantém erro limpo
        
        except Exception as e:
            raise RuntimeError(f"Erro ao obter classe em _get_classe: {e}")       
    
    #Move um registro para a tabela textos_treinamento
    def _insert_id_in_textos_treinamento(self, id: int, CodClasse: int, coduser: int) -> None:
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
            self.salva_log_alteracoesBll.insert_log_texto_treinamento([id], coduser, auto_commit=False)
                              
        except Exception as e:
            raise RuntimeError(f"Erro ao mover para textos_treinamento: {e}")

    #Move ou atualiza um ponto da collection train para a collection final no Qdrant."""
    def _insert_id_in_qdrant_final(self, id: int, embeddings: dict, CodClasse: int, classe:str) -> None:        
        #insere na colection final
        self.qdrant_utils.upinsert_id(collection_name=self.final_collection,
                                    id=id, 
                                    embeddings=embeddings,
                                    cod_classe=CodClasse
            )
            

    #Remove todos os registros relacionados ao idBase da tabela sugestao_textos_classificar.
    def _delete_sugestao_textos_classificar(self, idBase:int, idSimilar: int) -> None:                      
        try:
            query = f"DELETE FROM sugestao_textos_classificar WHERE IdBase = :IdBase and IdSimilar = :IdSimilar"
            self.session.execute(text(query), {"IdBase": idBase, "IdSimilar": idSimilar})        
        except Exception as e:
            raise RuntimeError(f"Erro ao deletar id {idSimilar} sugestao_textos_classificar: {e}")
        
                        
    def _regenerate_embedding_classificar(self, idBase:int,codClasse:int,classe:str) -> Optional[dict[str, Any]]:
        try:
            row = self.session.execute(text("""
                SELECT TxtTreinamento
                FROM textos_classificar
                WHERE id = :idBase
            """), {"idBase": idBase}).mappings().first()
            if not row:
                return None
            
            texto_treinamento = row["TxtTreinamento"]
            embedding_vector = self.embeddingsBll.generate_embedding(texto_treinamento,Id=idBase)

            if embedding_vector is None:
                raise RuntimeError(f"Erro ao gerar embedding para o ID {idBase}.")
            
            return {                
                "IdEncontrado": int(idBase),
                "Classe": classe,
                "CodClasse": codClasse,
                "Embedding": np.array(embedding_vector, dtype=np.float32)                
            }
        
        except Exception as e:
            raise RuntimeError(f"Erro ao recriar embedding para o ID {idBase}: {e}")    
        
    #Insere os IDs coletados para a coleção final do Qdrant.                    
    def _insert_ids_to_qdrant_final(self, lista_ids_mover: set, CodClasse:int, Classe: str, coduser:int) -> None:
        try:
            for id in lista_ids_mover:
                registro = self.qdrant_utils.get_id(id=id, collection_name=self.train_collection)
                if registro is None:
                    registro = self._regenerate_embedding_classificar(idBase=id, codClasse=CodClasse, classe=Classe)
                    if registro is None:                        
                        continue # aqui vai continuar pois ele pode ter sido movido previamente , logo não vou para a execução
                
                self._insert_id_in_qdrant_final(id, registro["Embedding"], CodClasse, Classe)
                self._insert_id_in_textos_treinamento(id=id, CodClasse=CodClasse, coduser=coduser)  # Move para textos_treinamento
                           
        except Exception as e:
            raise RuntimeError(f"Erro ao mover ids para o banco vetorial final: {e}") 
        
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
                
    def _delete_texto_classificar(self, id:int) -> None:                      
        try:
            query = f"DELETE FROM textos_classificar WHERE id = :id"
            self.session.execute(text(query), {"id": id})
            self.qdrant_utils.delete_id(collection_name=self.train_collection, id=id) #apaga do qdrant de treinamento   
        except Exception as e:
            raise RuntimeError(f"Erro ao deletar id {id} textos_classificar: {e}")
    
    #apaga o registro da tabela textos_treinamento
    def _delete_texto_treinamento(self, id:int) -> None:                      
        try:
            query = f"DELETE FROM textos_treinamento WHERE id = :id"
            self.session.execute(text(query), {"id": id})        
        except Exception as e:
            raise RuntimeError(f"Erro ao deletar id {id} textos_treinamento: {e}")
        
    #após todo o processamento de movimentação deve-se limpar os ids similares que são iguais para evitar duplicidade e economizar espaço
    def _limpa_ids_similares_iguais(self, lista_ids: set) -> None:
        try:
                if not lista_ids:
                    return                
                
                lista_ids = list(lista_ids)  # type: ignore

                param_ids = bindparam("ids", expanding=True)
                #define o tamanho do group concat para evitar truncamento com     GROUP_CONCAT(t.id ORDER BY t.id) AS TodosIDs em muitos ids duplicados
                self.session.execute(text("SET SESSION group_concat_max_len = 3000000"))                
                query = text("""
                            SELECT
                                Min(Id) as IdPrincipal,
                                TxtTreinamento,
                                GROUP_CONCAT(id ORDER BY id) AS IdsIguais
                            FROM textos_treinamento
                            WHERE id IN :ids
                            GROUP BY TxtTreinamento
                        """).bindparams(param_ids)
                
                
                result = self.session.execute(query, {"ids": lista_ids}).mappings().all()
                for row in result:
                    id = row["IdPrincipal"]
                    ids_iguais_str = row["IdsIguais"]
                    ids_iguais = [int(x) for x in ids_iguais_str.split(",")]
                    for id_igual in ids_iguais:
                        if id_igual != id:#só deve apagar se for diferente do id principal pois só pode ficar um
                            self._delete_texto_treinamento(id=id_igual)#apaga do banco de texto treinamento
                            self.qdrant_utils.delete_id(collection_name=self.final_collection, id=id_igual) #apaga do qdrant final
                    
        except Exception as e:
            raise RuntimeError(f"Erro ao limpar ids similares iguais: {e}")
        
    #caso o idBase não tenha mais registros na tabela sugestao_textos_classificar deve apagar o idBase da tabela textos_classificar
    def _limpa_ids_base_sem_similares(self, idBase:int) -> None:
        try:
           # tentar bloquear classificar idsimilares em o base base classificado
           # e aqui quando apagar verificar se só sobrou IdBase = IdBase tentar

            #verifica se o idBase ainda tem registros na tabela sugestao_textos_classificar
            query = f"SELECT COUNT(*) FROM sugestao_textos_classificar WHERE IdBase = :IdBase"
            result = self.session.execute(text(query), {"IdBase": idBase}).scalar()
            if (result or 0) == 0:
                #se não tiver mais registros deve apagar o idBase da tabela textos_classificar
                self._delete_texto_classificar(id=idBase)                
                self.qdrant_utils.delete_id(collection_name=self.train_collection, id=idBase) #apaga do qdrant de treinamento                   
                return
                

            # #verifica se o unico que sobrou como sugestão é ele mesmo ai pode apagar também
            # query = f"SELECT COUNT(*) FROM sugestao_textos_classificar WHERE (IdBase = :IdBase) and (IdSimilar = :IdBase)"
            # result = self.session.execute(text(query), {"IdBase": idBase}).scalar()
            # if result == 1:
            #     #se não tiver mais registros deve apagar o idBase da tabela textos_classificar
            #     self._delete_texto_classificar(id=idBase)                
            #     self.qdrant_utils.delete_id(collection_name=self.train_collection, id=idBase) #apaga do qdrant de treinamento                                   
                
        except Exception as e:
            raise RuntimeError(f"Erro ao limpar idBase sem similares {idBase}: {e}")
        
    #caso não encontrar o TxtTreinamento na tabela textos_classificar deve copiar de textos_treinamento para poder seguir o fluxo
    def _copia_texto_treinamento_para_textos_classificar(self, id:int) -> None:
        try:                  
            query = f"""
                select *from textos_treinamento where id = :id
            """
            row = self.session.execute(text(query), {"id": id}).mappings().first()
            if not row:
                raise RuntimeError(f"Erro em _copia_TextoTreinamento_para_textos_classificar O ID {id} não foi encontrado na tabela textos_treinamento para copiar para textos_classificar.")
            

            agora = datetime.now()
            query_insert = f"""
                INSERT INTO textos_classificar
                (id, DataEvento, Documento, CodClasse, UF, TxtDocumento, TxtTreinamento, QtdPalavras,
                TipoDefinicaoInicioTxt, ProcessadoNulo, PalavraIni, Indexado, BuscouIgual, BuscouColidente,DataHoraInsert,DataHoraEdit)
                SELECT id, DataEvento, Documento, :CodClasse, UF, TxtDocumento, TxtTreinamento, QtdPalavras,
                    TipoDefinicaoInicioTxt, ProcessadoNulo, PalavraIni, 1, 0, 0, :DataHoraInsert, :DataHoraEdit
                FROM textos_treinamento
                WHERE id = :id
                ON DUPLICATE KEY UPDATE
                    CodClasse = VALUES(CodClasse),
                    TxtTreinamento = VALUES(TxtTreinamento),
                    TxtDocumento = VALUES(TxtDocumento)
            """
            self.session.execute(text(query_insert), {"id": id, "CodClasse": row["CodClasse"], "DataHoraInsert": agora, "DataHoraEdit": agora})
            self._recreate_IdBase(id)
            
                              
        except Exception as e:
            raise RuntimeError(f"Erro ao mover para textos_treinamento: {e}")
    
    #verifica se o idBase existe na tabela textos_treinamento e no qdrantfinal pois primeiro deve ser classificado o IdBase = IdBase
    def _check_idBase_in_textos_treinamento(self, idBase:int) -> None:
        try:
            query = f"SELECT COUNT(*) FROM textos_treinamento WHERE id = :idBase"
            result = self.session.execute(text(query), {"idBase": idBase}).scalar()
            if (result or 0) == 0:
                raise RuntimeError(f"Erro: O IdBase {idBase} não foi encontrado na tabela textos_treinamento !\n Você deve primeiro informar o IdPrincipal para depois classificar os similares.") 
            
            if (self.qdrant_utils.get_id(id=idBase, collection_name=self.final_collection) == None):
                raise RuntimeError(f"Erro: O IdBase {idBase} não foi encontrado na coleção final!\n Você deve primeiro informar o IdPrincipal para depois classificar os similares.")


        except Exception as e:
            raise RuntimeError(f"Erro ao verificar idBase em textos_treinamento: {e}")        

    #Atualiza o campo ja classificado em sugestao_textos_classificar isso deve ser feito quando idbase = idbase para sinalizar para a interface
    # que o IdBase principal ja foi atualizado    
    def _update_ja_classificado_idbase(self, idBase:int):
        try:
            #atualiza o IbBase como ja classificado
            self.session.execute(text("Update sugestao_textos_classificar set JaClassificado = 1 where IdBase = :idBase"),{"idBase": idBase})
        except Exception as e:
            raise RuntimeError(f"Erro ao verificar idBase em _update_ja_classificado_idbase: {e}") 

    #Apaga o idBase da tabela sugestao_textos_classificar caso ele não tenha mais registros relacionados
    def _apaga_idbase_orfao(self, idBase:int):
        try:        
            #verifica se só tem o IdBase na tabela sugestao_textos_classificar para apagar o registro             
            query = text(f"""DELETE FROM sugestao_textos_classificar
                        WHERE IdBase = :idBase
                            AND (
                                SELECT COUNT(*)
                                FROM sugestao_textos_classificar
                                WHERE IdBase = :idBase        
                            ) = 1
                         """)
                 
            self.session.execute(query,{"idBase":idBase})
        except Exception as e:
            raise RuntimeError(f"Erro ao apagar idBase órfão em _apaga_idbase_orfao: {e}")

    #Main method to move suggested training texts based on similarity and class.
    #CodUser vem da interface do usuario
    #mover_com_colidencia serve para ignorar a verificação de colisão de classes caso o usuario queira forçar a movimentação
    def move_sugestao_treinamento(self, idBase: int, idSimilar: int, codClasse, coduser:int , mover_com_colidencia:bool=False) -> dict:
        try:    
            if not (self._check_reg_exists_in_sugestao_textos_classificar(idBase, idSimilar)):
                raise RuntimeError(f"Erro: Registro com IdBase {idBase} e IdSimilar {idSimilar} não encontrado em sugestao_textos_classificar")

            if (idBase != idSimilar):#faz isso para verificar se ja foi classificado o IdBase Principal
                self._check_idBase_in_textos_treinamento(idBase)

            #Bloco que procura o colidencias com idBase
            result = self._check_coliding_idBase(idBase=idBase,codClasse=codClasse,mover_com_colidencia=mover_com_colidencia)
            if (result is not None):
                return result
            
            #Bloco que procura o colidencias com idSimilar
            result = self._check_coliding_idSimilar(idSimilar=idSimilar,codClasse=codClasse,mover_com_colidencia=mover_com_colidencia)
            if (result is not None):
                return result
                   
            classe           = self._get_classe(codClasse)
            result           = self._get_ids_to_move(idBase, idSimilar, codClasse, classe, coduser)
            lista_similares  = result[0]# lista de ids a mover similares >= 98.5 e < 100            
            lista_iguais     = result[1] #lista de ids iguais a 100 que devem ser movidos para o treinamento

            if len(lista_iguais) > 0:#caso ids duplicados existirem mova apenas o primeiro ja que os outros são iguais
                id_duplicado = next(iter(lista_iguais))#pega o primeiro item do set
                self._insert_ids_to_qdrant_final(lista_ids_mover=[id_duplicado], CodClasse=codClasse, Classe=classe, coduser=coduser) 
                                            
            #Agora insere toda a lista de similares para o qdrant final e textos_treinamento
            self._insert_ids_to_qdrant_final(lista_similares, CodClasse=codClasse, Classe=classe, coduser=coduser)

            if (idBase == idSimilar):#caso for igual atualiza o ja classificado para a interface saber 
                self._update_ja_classificado_idbase(idBase)

            #deve apagar os registros duplicados da tabela sugestao_textos_classificar
            for item in lista_iguais:
                if (item != idBase):#não deve apagar o idBase pois pode ter outros similares para ele
                    self._delete_sugestao_textos_classificar(idBase=idBase, idSimilar=item)
                    self._delete_texto_classificar(id=item)
            
            #apaga os registros similares da tabela sugestao_textos_classificar
            for item in lista_similares:
                self._delete_sugestao_textos_classificar(idBase=idBase, idSimilar=item)       
                self._delete_texto_classificar(id=item) 

            self._limpa_ids_similares_iguais(lista_similares)
            self._limpa_ids_base_sem_similares(idBase=idBase)

            self._apaga_idbase_orfao(idBase)#apaga o idBase da tabela sugestao_textos_classificar caso ele não tenha mais registros relacionados
            
            self.session.commit()#grava as mudanças no banco de dados MariaDb idéia é que tudo seja feito numa transação só

            total_movido = len(lista_iguais) + len(lista_similares)
            sucessMessage = f"Movidos {total_movido} registros para treinamento e Modelo de IA"
            print_with_time(sucessMessage)
            return {
                "status": "OK",
                "mensagem": sucessMessage,
                "movidos": total_movido
            }

        except Exception as e:
            self.session.rollback()
            errorMessage = f"Erro! ao mover sugestões para treinamento:\n {e}"
            print_with_time(errorMessage)
            return {
                "status": "ERROR",
                "mensagem": errorMessage
            }
        finally:
            self.session.close()
