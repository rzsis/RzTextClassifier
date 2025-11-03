#sugere_textos_classificarBll.py
from ast import Dict, List
from calendar import c
import os
import math
from pathlib import Path
from typing_extensions import runtime
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, Filter, FieldCondition, MatchValue
from sqlalchemy import RowMapping, Sequence, text
from tqdm import tqdm
from sqlalchemy.orm import Session
from common import print_with_time, print_error, get_localconfig
from bll.classifica_textoBll import classifica_textoBll as classifica_textoBllModule
import bll.embeddingsBll as embeddingsBllModule
from bll.log_ClassificacaoBll import LogClassificacaoBll as LogClassificacaoBllModule
import gpu_utils as gpu_utilsModule
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule
import logger
from collections.abc import Sequence
import torch
from bll.indexa_textos_classificarBll import indexa_textos_classificarBll as indexa_textos_classificarBllModule
import time

##################################################
####################################################
# Idéia do algoritmo:
# 1 - Primeiro indexa todos os textos que faltam buscar similares no qdrant na base de treinamento que não foram classificados
# 2 - Buscar textos que não buscaram similar ainda primeiro encontrando os duplicados que são literalmente iguais e inserir em sugestao_textos_classificar
# 3 - Processar os textos duplicados primeiro para otimizar o processamento
# 4 - Buscar os textos que faltam buscar similares utilizando uma curva de similaridade baseada no tamanho do texto e nível de busca
# 5 - Inserir na tabela sugestao_textos_classificar os textos similares encontrados para sugerir classificação
####################################################    

class sugere_textos_classificarBll:
    def __init__(self, session: Session):
        """
        Inicializa a classe para indexação e detecção de textos similares usando Qdrant.
        Args:
            session (Session): Sessão SQLAlchemy para operações no banco.
        """
        try:
            from main import localconfig as localcfg
            self.session = session
            self.localconfig = localcfg
            self.config = localcfg.read_config()            
            self.limite_similares = 20
            self.similarity_threshold = 0.96
            self.min_similars = 3
            self.clusters = {} # Cache: {id_base: [{"id": id_similar, "score": score}, ...]}
            # Inicializa embeddings
            embeddingsBllModule.initBllEmbeddings(self.session)

            self.qdrant_utils = Qdrant_UtilsModule()
            self.textos_classificar_collection_name = self.qdrant_utils.get_collection_name("train")             
            self.qdrant_client = self.qdrant_utils.get_client()
            self.qdrant_utils.create_collection(self.textos_classificar_collection_name)
            self.classifica_textoBll = classifica_textoBllModule(embeddingsModule=embeddingsBllModule.bllEmbeddings, session=session)
            self.log_ClassificacaoBll = LogClassificacaoBllModule(session)
            self.logger = logger.log
            self.LimitePalavras = localcfg.get("max_length")    
            self.baseWhereSQLBuscarSimilarCorreto = f"""
                                    WHERE
                                        t.Indexado = true
                                    and t.Classificado = true                                    
                                    and t.TxtTreinamento IS NOT NULL and t.TxtTreinamento <> ''
                                    and t.Metodo in ('N','Q','M')
                                    and t.id not in (select IdBase from sugestao_textos_classificar)
                                    and t.id not in (select IdSimilar from sugestao_textos_classificar)                                    
                                    and t.QtdPalavras <= {self.LimitePalavras}
                                """   
                                         
            self.baseWhereSQLBuscarSimilar = f"""
                                    WHERE
                                        t.Indexado = true
                                    and t.TxtTreinamento is NOT NULL and t.TxtTreinamento <> ''
                                    and t.id not in (select IdBase from sugestao_textos_classificar)
                                    and t.id not in (select IdSimilar from sugestao_textos_classificar)
                                    and t.QtdPalavras <= {self.LimitePalavras}
                                    and (t.NivelBuscaSimilaridade is null or t.NivelBuscaSimilaridade <= 4)
                                """   
            self.gpu_utils = gpu_utilsModule.GpuUtils()
            self.limiteItensClassificar = localcfg.get("text_limit_per_batch")
            self.similares_inseridos = 0
                
        except Exception as e:
            raise RuntimeError(f"Erro ao inicializar sugestao_textos_classificarBll: {e}")

    #obtem a quantidade de textos pendentes falta buscar similar
    def _get_qtd_textos_falta_buscar_similar(self) -> int:
        try:
            query = f"""
                SELECT Count(t.id) AS TotalTextosPendentes
                FROM textos_classificar t
                {self.baseWhereSQLBuscarSimilar}                
                ORDER BY t.id
            """
            return self.session.execute(text(query)).mappings().all()[0]['TotalTextosPendentes']
        except Exception as e:
            raise RuntimeError(f"Erro ao obter _get_qtd_textos_falta_buscar_similar: {e}")
        

    #Obtem os textos que faltam buscar similares que são duplicados
    def _get_lista_textos_duplicados(self) -> Sequence[RowMapping]:
        try:
            query = f"""
                SELECT  t.id, 
                        t.TxtTreinamento AS Text,
                        Count(*) as QtdItens
                FROM textos_classificar t
                {self.baseWhereSQLBuscarSimilar}    
                and t.BuscouIgual = false
                group by t.TxtTreinamento
                having Count(*) >= {self.min_similars}            
                ORDER BY Count(*) DESC,t.id              
            """
            return self.session.execute(text(query)).mappings().all()
        except Exception as e:
            raise RuntimeError(f"Erro ao obter _get_lista_textos_duplicados: {e}")
        
    #obtem uma lista que tem todos os textos igual ao texto passado        
    def _get_texto_duplicado(self,id:int, texto:str):
        query = f"""
                SELECT  t.id, 
                        t.TxtTreinamento AS Text                                   
                FROM textos_classificar t
                    where  t.id <> {id}
                           and t.TxtTreinamento = '{texto}'
        """
        return self.session.execute(text(query)).mappings().all()
    
    def _get_QtdPalavras(self,texto:str) -> int:
        try:
            if (texto == None) or (texto.strip() == ''):
                return 0
            return len(texto.strip().split())
        except Exception as e:
            print_with_time(f"Erro em _get_QtdPalavras: {e}")
            return 0    
        

    #faz e processamento somente dos textos duplicados na base para otimizar o processamento
    def _processa_textos_duplicados(self,data):
        try:
            if len(data) == 0:
                print_with_time("Sem textos duplicados para processar")
                return
            
            qtd_inserido_similares = 0
            qtd_inserido = 0
            for item in tqdm(data,"Processando textos duplicados"):
                already_exists = (item['id'] in self.lista_sugestao_textos_classificar)
                if already_exists:
                    continue
                                    
                lista_marcar_duplicados = []
                lista_marcar_duplicados.append({"id":item["id"]})            

                lista_duplicados = self._get_texto_duplicado(item["id"],item["Text"])
                lista_insercao_duplicados = []
                for item_duplicado in lista_duplicados:
                    lista_marcar_duplicados.append({"id":item_duplicado["id"]})
                    lista_insercao_duplicados.append({
                            "IdEncontrado": item_duplicado["id"],
                            "Similaridade": 1
                        })
                    qtd_inserido += 1
                    
                self._insere_sugestao_textos_classificar(item["id"],lista_insercao_duplicados)
                self._mark_as_buscou_igual(lista_marcar_duplicados)
                
                #Agora vai criar um dado para buscar textos altamente similares para o texto base
                txtToFindSimilar = {  
                    'id': item["id"],
                    'QtdPalavras': self._get_QtdPalavras(item["Text"]),
                    'NivelBuscaSimilaridade': 0
                }    

                #agora vai procurar textos altamente similares para sugerir classificação
                lista_similares = self.get_similares(data=txtToFindSimilar,min_similarity=self.get_min_similarity(txtToFindSimilar['QtdPalavras'],1))
                self.insere_similares(item["id"], lista_similares,1)
                if lista_similares != None:
                    qtd_inserido_similares += len(lista_similares)

            print_with_time(f"Inseridos {qtd_inserido} textos duplicados + altamente similares {qtd_inserido_similares}")
        except Exception as e:
            print_with_time(f"Erro ao processar em _processa_textos_duplicados: {e}")

        
    #Obtem os textos que faltam buscar similares
    def _get_textos_falta_buscar_similar(self) -> Sequence[RowMapping]:
        try:
            query = f"""
                SELECT t.id, 
                    t.TxtTreinamento AS Text,
                    t.QtdPalavras,
                    t.NivelBuscaSimilaridade
                FROM textos_classificar t
                {self.baseWhereSQLBuscarSimilar}                
                ORDER BY t.id
                LIMIT {self.limiteItensClassificar}                
            """
            return self.session.execute(text(query)).mappings().all()
        except Exception as e:
            raise RuntimeError(f"Erro ao obter _get_textos_falta_buscar_similar: {e}")
    
    def get_min_similarity(self, qtdPalavras:int, nivelBusca:int) -> float:
        """
        Ajusta o nível mínimo de similaridade de acordo com o tamanho do texto
        e o nível de busca. Usa distância COSINE no Qdrant.
        """
        if (nivelBusca == 0):
            nivelBusca = 1

        p = max(0, float(qtdPalavras))
        base = 0.98
        reducao_base = [0.00, 0.04, 0.08, 0.13][nivelBusca - 1]

        # Fator de tamanho (curva suave)
        queda_curto = 0.30 * (1 - math.tanh(p / 70.0))
        subida_media = 0.15 * math.tanh((p - 200) / 180.0)
        ganho_longo = 0.08 * math.tanh((p - 600) / 200.0)
        fator_tamanho = 0.85 + queda_curto + subida_media + ganho_longo

        reducao = reducao_base * fator_tamanho
        similaridade = base - reducao

        # Piso: 0.87 apenas no Nível 4
        piso = 0.87 if nivelBusca == 4 else 0.85
        return round(max(similaridade, piso), 3)
                

    #faz a busca de similares e retorna a lista para inserir na sugestão de classificação
    # min_similarity = Caso eu quer uma similaridade mínima diferente da calculada, posso passar no parâmetro min_similarity        
    def get_similares(self,data: dict,  min_similarity:float = None) -> list: # type: ignore
        try:
            id              = data['id']  
            qtdPalavras     = data['QtdPalavras'] or 0
            nivelBusca      = data['NivelBuscaSimilaridade'] if data['NivelBuscaSimilaridade'] is not None else 0
            if min_similarity is None:
                min_similarity  = self.get_min_similarity(qtdPalavras, nivelBusca)


            id_found        = self.qdrant_utils.get_id(id=id, collection_name=self.textos_classificar_collection_name)                    
            if (id_found == None):
                return None # type: ignore
                    
            result          = self.classifica_textoBll.check_embedding_colliding(query_embedding= id_found["Embedding"],
                                                                        collection_name=self.textos_classificar_collection_name,
                                                                        id_a_classificar= None,
                                                                        TabelaOrigem="C",
                                                                        itens_limit=100,
                                                                        gravar_log=False,
                                                                        min_similarity=min_similarity)
            if (result == None) or (result.ListaSimilaridade == None):
                return None # type: ignore

            return [item.__dict__ for item in result.ListaSimilaridade if item.IdEncontrado not in self.lista_sugestao_textos_classificar] # type: ignore
        except Exception as e:
            print_with_time(f"erro em get_similares {e} ")
            
    #obtem uma lista de sugestao_textos_classificar gerando uma lista dupla com IdSimilar e IdBase igual
    #pois uma vez um IdInserido ele não deve ser considerado similar a outro logo não deve ser inserido novamente
    def _get_list_sugestao_textos_classificar(self): # type: ignore
        try:
            query = f"""
                SELECT t.IdBase as Id
                FROM sugestao_textos_classificar t
                Group by t.IdBase                         
                Union
                SELECT t.IdSimilar as Id
                FROM sugestao_textos_classificar t
                Group by t.IdSimilar                                            
            """
            rows = self.session.execute(text(query)).mappings().all()
            return {row["Id"] for row in rows}

        except Exception as e:
            raise RuntimeError(f"Erro ao obter get_list_sugestao_textos_classificar: {e}")
                
    #insere as sugestões de textos similares na tabela sugestao_textos_classificar
    def _insere_sugestao_textos_classificar(
        self, 
        id_texto: int, 
        similars: list[dict], 
    ):
        try:
            if self.lista_sugestao_textos_classificar is not None:
                self.lista_sugestao_textos_classificar.add(id_texto)

            # Monta os valores de forma segura
            valores = []
            for similar in similars:
                id_similar = similar.get('IdEncontrado')
                similaridade = (similar.get('Similaridade') or 0) * 100

                if self.lista_sugestao_textos_classificar is not None:
                    self.lista_sugestao_textos_classificar.add(id_similar)

                valores.append(f"({id_texto}, {id_similar}, {similaridade}, NOW())")

            if valores:
                query = f"""
                    INSERT IGNORE INTO sugestao_textos_classificar (IdBase, IdSimilar, Similaridade, DataHora)
                    VALUES {', '.join(valores)}
                """
                self.session.execute(text(query))
                self.session.commit()

        except Exception as e:
            self.session.rollback()
            raise RuntimeError(f"Erro ao inserir sugestao_textos_classificar: {e}")


    #depois de processado marca como BuscouIgual a lista que foi processada
    def _mark_as_buscou_igual(self, data: list):
        try:
            ids_to_update = [row['id'] for row in data]
            query = """
                UPDATE textos_classificar
                SET 
                    BuscouIgual = true                   
                WHERE id IN :ids;
            """
            self.session.execute(text(query), {"ids": tuple(ids_to_update)})
            self.session.commit()            
        except Exception as e:
            print_with_time(f"Erro ao marcar textos como buscou similar em lote: {e}")
            self.session.rollback()

    #Incrementa o nível de similaridade para os textos processados
    def _inc_nivel_similaridade(self, data: list):
        try:
            ids_to_update = [row['id'] for row in data]
            query = """
                UPDATE textos_classificar
                SET                     
                    NivelBuscaSimilaridade = COALESCE(NivelBuscaSimilaridade, 0) + 1
                WHERE id IN :ids;
            """
            self.session.execute(text(query), {"ids": tuple(ids_to_update)})
            self.session.commit()            
        except Exception as e:
            print_with_time(f"Erro ao marcar textos como buscou similar em lote: {e}")
            self.session.rollback()            


    #Obtem a quantidade de textos pendentes a classificar
    def _get_qtd_textos_pendentes_classificar(self) -> int:
        try:
            query = f"""
                SELECT Count(t.id) AS TotalTextosPendentes
                FROM textos_classificar t
                {self.baseWhereSQLClassificar}
                ORDER BY t.id
            """
            return self.session.execute(text(query)).mappings().all()[0]['TotalTextosPendentes']
        except Exception as e:
            raise RuntimeError(f"Erro ao obter _get_qtd_textos_pendentes_classificar: {e}")

    #insere os similares encontrados caso tenha mais que o mínimo definido
    def insere_similares(self, id_texto:int, lista_similares:list, min_similars:int):
        #caso tiver mais que X amostras de similares insere para sugerir para classificar    
        try:
            if (lista_similares != None) and (len(lista_similares) >= min_similars):
                already_exists  = False
                for similar in lista_similares:                         
                    already_exists = (similar['IdEncontrado'] in self.lista_sugestao_textos_classificar)
                    if already_exists:
                            break

                    if (already_exists == False):#caso o registro não existir insere
                        self._insere_sugestao_textos_classificar(id_texto, lista_similares)                                                    
                        self.similares_inseridos += 1

        except Exception as e:
           print_with_time(f"Erro ao inserir similares para texto id {id_texto}: {e}")   

    #pega todos os textos pendentes que o sistema não buscou similar procura similares agrupa por similaridade
    #para depois o usuario sugerir classificações
    def sugere_textos_para_classificar(self) -> dict:
        try:
            #primeiro deve indexar tudo no qdrant para depois fazer a busca
            inicio = time.time()
            indexa_textos_classificarBll = indexa_textos_classificarBllModule(self.session)
            indexa_textos_classificarBll.indexa_textos_classificar()

            print_with_time(f"Iniciando busca de textos duplicados...")
            self.lista_sugestao_textos_classificar = self._get_list_sugestao_textos_classificar()            
            data = self._get_lista_textos_duplicados()
            self._processa_textos_duplicados(data)

            print_with_time(f"Iniciando busca de textos similares...")
            data = self._get_textos_falta_buscar_similar()
            self.lista_sugestao_textos_classificar = self._get_list_sugestao_textos_classificar()

            if not data:
                sucessMessage = "Nenhum texto similar restante para classificar"
                print_with_time(sucessMessage)
                return {
                    "status": "OK",
                    "processados": sucessMessage,
                    "restante": f"0"
                }
            
            self.similares_inseridos = 0
            for row in tqdm(data, desc="Processando textos para busca de similares"):
                try:
                    lista_similares = self.get_similares(data=row)
                    self.insere_similares(row['id'], lista_similares, self.min_similars)
                
                except Exception as e:
                    print_with_time(f"Erro ao buscar similar de texto id {row['id']}: {e}")

            self._inc_nivel_similaridade(data)
            self.gpu_utils.clear_gpu_cache()

            tempo_decorrido_min = (time.time() - inicio) / 60          
            sucessMessage = f"Inseridos {self.similares_inseridos} sugestões de textos similares, Tempo decorrido: {tempo_decorrido_min:.2f} minutos"

            print_with_time(sucessMessage)
            return {
                "status": "OK",
                "mensagem": sucessMessage,
                "restante": f"{self._get_qtd_textos_falta_buscar_similar()}"
            }
        except Exception as e:
            errorMessage = f"Erro ao processar textos para busca de similares: {e}"
            print_error(errorMessage)
            return {
                "status": "ERROR",
                "processados": errorMessage,
                "restante": ""
            }
    