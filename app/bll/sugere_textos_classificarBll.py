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
# Sugere textos para classificar baseado em textos similares encontrados no Qdrant
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
                                    and (t.NivelBuscaSimilaridade is null Or t.NivelBuscaSimilaridade <= 5)
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
    def _get_lista_textos_duplicados(self) -> list[dict]:
        try:
            #define o tamanho do group concat para evitar truncamento com     GROUP_CONCAT(t.id ORDER BY t.id) AS TodosIDs em muitos ids duplicados
            self.session.execute(text("SET SESSION group_concat_max_len = 3000000"))
        
            query = f"""
                SELECT  Min(t.id) as id, 
                        t.TxtTreinamento AS Text,
                        Count(*) as QtdItens,
                        GROUP_CONCAT(t.id ORDER BY t.id) AS TodosIDs
                FROM textos_classificar t
                {self.baseWhereSQLBuscarSimilar}    
                and t.BuscouIgual = false
                group by t.TxtTreinamento 
                Having Count(*) >= 3               
                ORDER BY Count(*) DESC,t.id           
            """
            rows = self.session.execute(text(query)).mappings().all()

            if not rows:
                return []

            grupos = []

            for row in rows:
                texto_atual = row['Text']
                id_atual = row['id']
                ids_duplicados = [int(x) for x in row['TodosIDs'].split(',')]    
                grupos.append({
                        "IdBase": id_atual,
                        "Text": texto_atual,
                        "IdsIguais": ids_duplicados
                })

            return grupos

        except Exception as e:
            raise RuntimeError(f"Erro ao obter _get_lista_textos_duplicados: {e}")
        
    #obtem uma lista que tem todos os textos igual ao texto passado        
    def _get_texto_duplicado(self,id:int, texto:str):
        query = f"""
                SELECT  t.id                                 
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

    #processa os textos duplicados encontrados inserindo na sugestao_textos_classificar 
    def _processa_textos_duplicados(self,data):
        try:
            if len(data) == 0:
                print_with_time("Sem textos duplicados para processar")
                return
                           
            qtd_inserido_similares = 0
            qtd_inserido = 0
            for item in tqdm(data,"Processando textos duplicados"):
                already_exists = (item['IdBase'] in self.lista_sugestao_textos_classificar)
                if already_exists:  
                    continue
                                    
                lista_marcar_duplicados = []
                lista_insercao_duplicados = []                

                lista_marcar_duplicados.append({"id":item["IdBase"]})
                #insere ele mesmo na lista para ter IdBase,IdBase            
                lista_insercao_duplicados.append({
                            "IdEncontrado": item["IdBase"],
                            "Similaridade": 1
                        })
                
                for item_duplicado in item["IdsIguais"]:                    
                    lista_marcar_duplicados.append({"id":item_duplicado})
                    lista_insercao_duplicados.append({
                            "IdEncontrado": item_duplicado,
                            "Similaridade": 1
                        })
                    qtd_inserido += 1
                    

                self._insere_sugestao_textos_classificar(item["IdBase"],lista_insercao_duplicados)
                self._mark_as_buscou_igual(lista_marcar_duplicados)
                self.session.commit()   
                
                #Agora vai criar um dado para buscar textos altamente similares para o texto base
                txtToFindSimilar = {  
                    'id': item["IdBase"],
                    'QtdPalavras': self._get_QtdPalavras(item["Text"]),
                    'NivelBuscaSimilaridade': 0
                }    

                #agora vai procurar textos altamente similares para sugerir classificação
                lista_similares = self.get_similares(data=txtToFindSimilar,
                                                     min_similarity=self.get_min_similarity(txtToFindSimilar['QtdPalavras'],1),
                                                     itens_limit=500)
                
                self._insere_similares(item["IdBase"], lista_similares, 1)
                if lista_similares != None:
                    qtd_inserido_similares += len(lista_similares)
                    for similar in lista_similares:
                        lista_marcar_duplicados.append({"id":similar['IdEncontrado']}) 

                    self._mark_as_buscou_igual(lista_marcar_duplicados)
                    self.session.commit()

            print_with_time(f"Inseridos {qtd_inserido} textos duplicados + altamente similares {qtd_inserido_similares}")
        except Exception as e:
            print_with_time(f"Erro ao processar em _processa_textos_duplicados: {e}")


    #Obtem os textos que faltam buscar similares ou seja textos que ainda não foram inseridos em sugestao_textos_classificar
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
    
        """   Curva de similaridade mínima baseada em quantidade de palavras e nível de busca
          Palavras |     N1     N2     N3     N4     N5
        --------     |--------------------------------------
             0  |  0.980  0.960  0.940  0.920  0.900
            25  |  0.980  0.960  0.940  0.920  0.900
            50  |  0.981  0.961  0.941  0.921  0.901
            75  |  0.982  0.962  0.942  0.922  0.901
            100 |  0.983  0.962  0.942  0.922  0.902
            125 |  0.984  0.963  0.943  0.923  0.902
            150 |  0.985  0.964  0.944  0.923  0.903
            175 |  0.986  0.965  0.945  0.924  0.903
            200 |  0.987  0.966  0.946  0.925  0.904
            225 |  0.987  0.966  0.946  0.925  0.904
            250 |  0.988  0.967  0.947  0.926  0.905
            275 |  0.988  0.968  0.947  0.926  0.905
            300 |  0.989  0.968  0.948  0.927  0.906
            325 |  0.989  0.969  0.948  0.927  0.906
            350 |  0.990  0.969  0.949  0.928  0.906
            375 |  0.990  0.970  0.949  0.928  0.907
            400 |  0.990  0.970  0.950  0.928  0.907
            425 |  0.991  0.971  0.950  0.929  0.907
            450 |  0.991  0.971  0.951  0.929  0.908
            475 |  0.991  0.971  0.951  0.929  0.908
            500 |  0.992  0.972  0.951  0.930  0.908
            525 |  0.992  0.972  0.952  0.930  0.908
            550 |  0.992  0.972  0.952  0.930  0.909
            575 |  0.992  0.973  0.952  0.930  0.909
            600 |  0.993  0.973  0.953  0.931  0.909
            625 |  0.993  0.973  0.953  0.931  0.909
            650 |  0.993  0.974  0.953  0.931  0.909
            675 |  0.993  0.974  0.953  0.931  0.910
            700 |  0.993  0.974  0.954  0.932  0.910
            725 |  0.993  0.974  0.954  0.932  0.910
            750 |  0.994  0.974  0.954  0.932  0.910
            775 |  0.994  0.975  0.954  0.932  0.910
            800 |  0.994  0.975  0.954  0.932  0.910
            825 |  0.994  0.975  0.955  0.932  0.910
            850 |  0.994  0.975  0.955  0.933  0.911
            875 |  0.994  0.975  0.955  0.933  0.911
            900 |  0.994  0.975  0.955  0.933  0.911
            925 |  0.994  0.975  0.955  0.933  0.911
            950 |  0.994  0.976  0.955  0.933  0.911
            975 |  0.994  0.976  0.955  0.933  0.911
            1000 |  0.995  0.976  0.956  0.933  0.911
            1024 |  0.995  0.976  0.956  0.933  0.911 
    """
        
    # Calcula o nível mínimo de similaridade (COSINE) aproximando a tabela de 5 níveis fornecida.    
    #Usa uma função analítica suave baseada em tanh(), com piso mínimo de 0.90.        
    def get_min_similarity(self, qtdPalavras:int, nivelBusca:int) -> float:
        # Limitar faixa de entrada
        p = max(0, min(float(qtdPalavras), 1024))
        nivel = max(1, min(nivelBusca, 5))
        
        # Base inicial (nível 1 = 0.98, reduz 0.02 a cada nível)
        base = 0.98 - (nivel - 1) * 0.02
        
        # Ganho máximo conforme o nível
        # Níveis mais altos têm crescimento um pouco menor
        max_gain = 0.02 - (nivel - 1) * 0.001
        
        # Curva suave: sobe até ~300–500 palavras e estabiliza
        ganho = max_gain * math.tanh(p / 500)
        
        # Similaridade calculada
        similaridade = base + ganho
        
        # Piso mínimo (0.90) e teto lógico (1.0)
        piso = 0.90
        teto = 1.0
        
        return round(max(min(similaridade, teto), piso), 3)
                

    #faz a busca de similares e retorna a lista para inserir na sugestão de classificação
    #min_similarity = Caso eu quer uma similaridade mínima diferente da calculada, posso passar no parâmetro min_similarity        
    def get_similares(self,data: dict,  min_similarity:float = None,itens_limit:int = 100) -> list: # type: ignore
        try:
            id              = data['id']  
            qtdPalavras     = data['QtdPalavras'] or 0
            nivelBusca      = data['NivelBuscaSimilaridade'] if data['NivelBuscaSimilaridade'] is not None else 0
            if min_similarity is None:
                min_similarity  = self.get_min_similarity(qtdPalavras, nivelBusca)


            id_found        = self.qdrant_utils.get_id(id=id, collection_name=self.textos_classificar_collection_name)                    
            if (id_found == None):
                return None # type: ignore
                    
            result          = self.classifica_textoBll.get_similarity_list(query_embedding= id_found["Embedding"],
                                                                        collection_name=self.textos_classificar_collection_name,
                                                                        id_a_classificar= None,
                                                                        TabelaOrigem="C",
                                                                        itens_limit=itens_limit,
                                                                        gravar_log=False,
                                                                        min_similarity = min_similarity,
                                                                        exclusion_list = self.lista_sugestao_textos_classificar)
            if (result == None) or (result.ListaSimilaridade == None):
                return None # type: ignore
            
            lista_similares = [item.__dict__ for item in result.ListaSimilaridade if item.IdEncontrado not in self.lista_sugestao_textos_classificar] # type: ignore
            return lista_similares
        except Exception as e:
            print_with_time(f"erro em get_similares {e} ")
            
    #obtem uma lista de sugestao_textos_classificar ja inseridos no banco gerando uma lista dupla com IdSimilar e IdBase igual
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
                
    def _insere_sugestao_textos_classificar_duplicados(self, id_texto: int, similars: list[dict]) -> None:
            """
            Insere sugestões de textos similares na tabela `sugestao_textos_classificar`.
            Um INSERT por chamada (por grupo), com COMMIT imediato.
            
            Args:
                id_texto (int): IdBase
                similars (list[dict]): [{'IdEncontrado': int, 'Similaridade': float}, ...]
            """
            if not similars:
                return

            try:
                if self.lista_sugestao_textos_classificar is not None:
                    self.lista_sugestao_textos_classificar.add(id_texto)

                params = []
                values = []

                for idx, sim in enumerate(similars):
                    id_similar = sim['IdEncontrado']
                    similaridade = int((sim.get('Similaridade') or 0.0) * 100)  # 0-100

                    # Atualiza cache
                    if (self.lista_sugestao_textos_classificar is not None and id_similar not in self.lista_sugestao_textos_classificar):
                        self.lista_sugestao_textos_classificar.add(id_similar)

                    # Bind parameters
                    params.append({
                        f'id_base_{idx}': id_texto,
                        f'id_similar_{idx}': id_similar,
                        f'similaridade_{idx}': similaridade
                    })
                    values.append(f"(:id_base_{idx}, :id_similar_{idx}, :similaridade_{idx}, NOW())")

                query = f"""
                    INSERT IGNORE INTO sugestao_textos_classificar 
                        (IdBase, IdSimilar, Similaridade, DataHora)
                    VALUES {', '.join(values)}
                """

                # Junta todos os params em um único dict
                flat_params = {}
                for p in params:
                    flat_params.update(p)

                self.session.execute(text(query), flat_params)
                self.session.commit()

            except Exception as e:
                self.session.rollback()
                raise RuntimeError(f"Erro ao inserir sugestões para IdBase {id_texto}: {e}")

    #insere as sugestões de textos similares na tabela sugestao_textos_classificar
    def _insere_sugestao_textos_classificar(self, id_texto: int, similars: list[dict]):        
        try:
            if (self.lista_sugestao_textos_classificar is not None) and not (id_texto in self.lista_sugestao_textos_classificar):
                self.lista_sugestao_textos_classificar.add(id_texto)

            # Monta os valores de forma segura
            valores = []
            for similar in similars:
                id_similar = similar.get('IdEncontrado')
                similaridade = (similar.get('Similaridade') or 0) * 100

                if (self.lista_sugestao_textos_classificar is not None) and not (id_similar in self.lista_sugestao_textos_classificar):
                    self.lista_sugestao_textos_classificar.add(id_similar)

                valores.append(f"({id_texto}, {id_similar}, {similaridade}, NOW())")

            if valores:
                query = f"""
                    INSERT IGNORE INTO sugestao_textos_classificar (IdBase, IdSimilar, Similaridade, DataHora)
                    VALUES {', '.join(valores)}
                """
                self.session.execute(text(query))

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
    def _insere_similares(self, id_texto:int, lista_similares:list, min_similares:int = 3):
        #caso tiver mais que X amostras de similares insere para sugerir para classificar    
        try:
            if (lista_similares != None) and (len(lista_similares) >= min_similares):
                already_exists  = False
                for similar in lista_similares:                         
                    already_exists = (similar['IdEncontrado'] in self.lista_sugestao_textos_classificar)
                    if already_exists:
                        continue
                    
                    self._insere_sugestao_textos_classificar(id_texto, lista_similares)                                                    
                    self.similares_inseridos += len(lista_similares)

        except Exception as e:
           print_with_time(f"Erro ao inserir similares para texto id {id_texto}: {e}")   

    #atualiza os campos DataEvento e QtdPalavras na tabela sugestao_textos_classificar para aumentar a performance nas consultas
    #faz somente aqui pois pegar essa informação na hora da consulta é muito custoso
    def _update_textos_classificar(self):
        try:
            #atualiza data de evento e quantidade de palavras na tabela sugestao_textos_classificar
            query = """
              Update sugestao_textos_classificar stc 
                inner join textos_classificar tc on tc.id  = stc.IdSimilar 
                set stc.DataEvento = tc.DataEvento,
                stc.QtdPalavras = tc.QtdPalavras
                WHERE 
                stc.QtdPalavras is null or stc.QtdPalavras = 0
                or stc.DataEvento is null                 
            """
            self.session.execute(text(query))

            #atualiza a quantidade de similares encontrados para cada IdBase para o Id com mais semelhanças ficar no topo e não sumir do nada
            query = """
                UPDATE sugestao_textos_classificar AS stc
                JOIN (
                    SELECT IdBase, COUNT(*) AS Qtd
                    FROM sugestao_textos_classificar
                    GROUP BY IdBase
                ) AS agg ON stc.IdBase = agg.IdBase
                SET stc.QtdSimilares = agg.Qtd;
            """

            self.session.execute(text(query))
        except Exception as e:
            print_with_time(f"Erro em _update_textos_classificar: {e}")
            self.session.rollback()

    #para depois o usuario sugerir classificações
    #NivelBuscaSimilar = 0 pois assim ele só verifica o nivel atual caso maior ele vai incrementando e buscando mais similares até o nivel 5
    def sugere_textos_para_classificar(self,NivelBuscaSimilar:int = 0,ContadorEntrada = 0) -> dict:
        try:
            #primeiro deve indexar tudo no qdrant para depois fazer a busca
            inicio = time.time()
            indexa_textos_classificarBll = indexa_textos_classificarBllModule(self.session)
            indexa_textos_classificarBll.indexa_textos_classificar()

            print_with_time(f"Iniciando busca de textos duplicados...")
            self.lista_sugestao_textos_classificar = self._get_list_sugestao_textos_classificar()#obtem a lista atual ja inserida em sugestao_textos_classificar ja inserido no banco
            data = self._get_lista_textos_duplicados()
            self._processa_textos_duplicados(data)                     

            print_with_time(f"Iniciando busca de textos similares Nivel {ContadorEntrada}...")
            data = self._get_textos_falta_buscar_similar()
            self.lista_sugestao_textos_classificar = self._get_list_sugestao_textos_classificar()#obtem denovo a lista atual ja inserida em sugestao_textos_classificar 

            if not data:
                sucessMessage = "Nenhum texto similar restante para classificar"
                print_with_time(sucessMessage)
                return {
                    "status": "OK",
                    "processados": sucessMessage,
                    "restante": f"0"
                }
            
            self.similares_inseridos = 0
            for row in tqdm(data, desc="Processando textos para buscar similares"):
                try:
                    lista_similares = self.get_similares(data=row,itens_limit=100)
                    self._insere_similares(row['id'], lista_similares, 3)
                
                except Exception as e:
                    print_with_time(f"Erro ao buscar similar de texto id {row['id']}: {e}")

            self._inc_nivel_similaridade(data)
            self.gpu_utils.clear_gpu_cache()
            self._update_textos_classificar()
            self.session.commit()
            tempo_decorrido_min = (time.time() - inicio) / 60          
            sucessMessage = f"Inseridos {self.similares_inseridos} sugestões de textos similares, Tempo decorrido: {tempo_decorrido_min:.2f} minutos"
            itens_restantes = self._get_qtd_textos_falta_buscar_similar()
            
            #aqui caso faltem itens e o nivel de busca seja menor que 5 faz uma nova chamada recursiva para buscar mais similares
            if (itens_restantes > 0) and ((NivelBuscaSimilar > 0) and (ContadorEntrada <= 5)): 
                print_with_time(f"{sucessMessage} no nível {ContadorEntrada}, buscando próximos níveis...")                   
                self.sugere_textos_para_classificar(NivelBuscaSimilar, ContadorEntrada+1)        
                itens_restantes = self._get_qtd_textos_falta_buscar_similar()
                
            print_with_time(sucessMessage)
            return {
                "status": "OK",
                "mensagem": sucessMessage,
                "restante": f"{itens_restantes}"
            }
        except Exception as e:
            errorMessage = f"Erro ao processar textos para busca de similares: {e}"
            print_error(errorMessage)
            return {
                "status": "ERROR",
                "processados": errorMessage,
                "restante": ""
            }
    