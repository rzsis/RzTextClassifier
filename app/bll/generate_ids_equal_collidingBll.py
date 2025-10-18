# GenerateIdsIguaisCollindgsBLL.py
import os
from pathlib import Path
import numpy as np
from sqlalchemy import text
from tqdm import tqdm
from sqlalchemy.orm import Session
from common import print_with_time, print_error, get_localconfig
from bll.idIguaisBll import IdIguaisBll as IdIguaisBllModule
from bll.embeddings_generateBll import Embeddings_GenerateBll
from bll.idCollidingBll import IdCollidingBll as IdCollidingBllModule
from bll.embeddingsBll import EmbeddingsBll
from bll.classifica_textoBll import classifica_textoBll
import dbClasses.idIguais as idIguaisModule
import dbClasses.idsColidentes as idCollidingModule
from qdrant_utils import Qdrant_Utils as Qdrant_UtilsModule
import qdrant_utils

class GenerateIdsIguaisCollindgs:
    def __init__(self, session: Session, localcfg):
        self.session = session
        self.localconfig = localcfg
        self.config = localcfg.read_config()
        self.dataset_path = Path(self.config["dataset_path"])
        self.itens_limit = 50  # Number of nearest neighbors to search
        self.id_iguais_bll = IdIguaisBllModule(session)
        self.id_colliding_bll = IdCollidingBllModule(session)
        self.generate_embeddings = Embeddings_GenerateBll(session)
        self.embeddings_handler = EmbeddingsBll()
        self.qdrant_utils = Qdrant_UtilsModule()
        self.qdrant_client = self.qdrant_utils.get_client()
        self.collection_name = self.qdrant_utils.get_collection_name("final")
        self.classifica_texto = classifica_textoBll(self.embeddings_handler, session)
        LimitePalavras = localcfg.get("max_length")        
        self.baseWhereSQL = f"""
                                WHERE LENGTH(TRIM(t.TxtTreinamento)) > 0
                                AND t.CodClasse IS NOT NULL
                                and not t.id in (Select id from idsduplicados)                                
                                and t.Indexado = true 
                                and QtdPalavras <= {LimitePalavras}                                                           
                            """  # Filtra textos não vazios e não nulos, não duplicados, não iguais e não indexados
        self.limiteItensClassificar = self.localconfig.get("text_limit_per_batch")

    #get the count of data to be processed
    def _get_data_to_process(self,auxFilter) -> int:
        try:
            query = f"""
                SELECT COUNT(t.id) AS TotalTextosPendentes
                FROM textos_treinamento t    
                {self.baseWhereSQL} 
                {auxFilter}               
            """            
            return self.session.execute(text(query)).mappings().all()[0]['TotalTextosPendentes']

        except Exception as e:
            raise RuntimeError(f"Erro ontendo _get_Textos_Pendentes: {e}")
        
    #faz a consulta no banco de dados para obter os dados a serem processados
    def _fetch_data(self, auxFilter) -> list:        
        query = f"""
            SELECT MIN(t.id) AS Id,
                   c.CodClasse,
                   c.Classe,
                   t.TxtTreinamento AS Text,
                   COUNT(t.id) AS QtdItens
            FROM textos_treinamento t
                    INNER JOIN classes c ON c.CodClasse = t.CodClasse
                    {self.baseWhereSQL}
                    {auxFilter}
            GROUP BY t.TxtTreinamento, t.CodClasse, c.Classe
            Order by COUNT(t.id) DESC
            limit {self.limiteItensClassificar}
        """

        # Busca dados do banco de dados
        try:
            result = self.session.execute(text(query)).mappings().all()
            dados = [dict(row) for row in result]
                        
            return dados
        except Exception as e:
            raise RuntimeError(f"Erro executando consulta no banco de dados: {e}")
        

    def _genetare_ids_colliding(self) -> str:
        """
        Process a dataset to find colliding items (high similarity, different classes) and save to database.
        """
        similarity_threshold_colliding = 0.95
        # Load data from database
        auxFilter = " and BuscouIgual = true and BuscouColidente = false "
        data = self._fetch_data(auxFilter)
        
        # Process similarities  
        ids_Colidentes_Atuais = self.id_colliding_bll.get_all_ids_colidentes()                     
        ids_collidentes_para_inserir = []
        ids_verificados = []
        print_with_time(f"Processando {len(data)} registros para colisões...")
        for  item in tqdm(data, desc="Buscando Colidentes"):          
            id_tram = item["Id"]
            ids_verificados.append(id_tram)

            # Retrieve query embedding from Qdrant            
            result = self.qdrant_utils.get_id(id_tram, self.collection_name) or None
            if result is None:
                print_with_time(f"Aviso: Id {id_tram} não encontrado no Qdrant, pulando")
                continue
                
            if result["Embedding"] is None:
                print_with_time(f"Aviso: Embedding do id: {id_tram} vazio, pulando")
                continue
                
            query_embedding = result["Embedding"] 
            
            # Busca texto similares usando classifica_textoBll
            try:
                result = self.classifica_texto.search_similarities(
                    query_embedding=query_embedding,
                    collection_name=self.collection_name,
                    id_a_classificar=id_tram,
                    TabelaOrigem="textos_treinamento",
                    itens_limit=self.itens_limit,
                    gravar_log=False,
                    min_similarity=similarity_threshold_colliding
                )
                results = result.ListaSimilaridade or []
            except Exception as e:
                print_with_time(f"Erro ao buscar similares para Id {id_tram}: {e}")
                continue

            inseriuIdBase = False
            for similar_item in results:
                sim = similar_item.Similaridade or 0
                neighbor_id = similar_item.IdEncontrado
                neighbor_cod_classe = similar_item.CodClasse
                if neighbor_id == id_tram:
                    continue

                # Adiciona na lista de ids colidentes se a classe for diferente e a similaridade maior que o limiar
                if item["CodClasse"] != neighbor_cod_classe:                   
                    # Check if neighbor_id is already in idsColidentesAtuais
                    already_exists =  (neighbor_id in ids_Colidentes_Atuais) or (id_tram in ids_Colidentes_Atuais)

                    if not already_exists:
                        inseriuIdBase  = True
                        ids_Colidentes_Atuais.add(neighbor_id)  # Add to existing to avoid duplicates in this run                                                
                        ids_collidentes_para_inserir.append(
                            idCollidingModule.IdsColidentes(
                                Id=id_tram,
                                IdColidente=neighbor_id,
                                semelhanca=float((sim or 0) * 100),
                            )
                        )           
            #fim for insere lista de colidentes

            if inseriuIdBase:
                ids_Colidentes_Atuais.add(id_tram)  #Insere o Id da tramitacao somente no final pois ele pode ser colidente de varios outros


        # Insert idscolidentes into database
        try:
            itens_inseridos = self.id_colliding_bll.commit_lista(ids_collidentes_para_inserir)            
            if itens_inseridos > 0:
                print_with_time(f"Inserido no banco em IdsColidentes: {itens_inseridos}")
            else:
                print_with_time("Nenhum IdsColidentes inserido, lista vazia ou erro na inserção.")

        except Exception as e:
            print_error(f"Erro ao inserir IdsColidentes no banco: {e}")
            raise

        try:
            # Atualiza BuscouColidente para os itens processados para não processa-los novamente
            self.id_colliding_bll.set_buscou_colidente(ids_verificados)
        except Exception as e:
            print_error(f"Erro ao atualizar BuscouColidente: {e}")
            raise
        
        result = f"Registros colidentes inseridos: {len(ids_collidentes_para_inserir)}, faltam {self._get_data_to_process(auxFilter)} textos para processar."
        print_with_time(result)

        return result

    #Processa todos os textos contidos em textos_treinamento para encontrar ids com uma similaridade alta e mesma classe
    def _generate_ids_equal(self):
        """
        Process a dataset to find similar items (high similarity, same class) and save to database.
        """
        similarity_threshold_equal = 0.985
        # Load data from database
        auxFilter = """ and BuscouIgual = false 
                        and not t.id in (Select id from idsiguais)
                        and not t.id in (Select idIgual from idsiguais)
                    """
        
        data = self._fetch_data(auxFilter)
        if len(data) == 0:
            return  {
                "status": "Sucesso",
                "message":  "Nenhum registro pendente para procurar ids iguais.",
                }
      
        # Process similarities
        keep_indices = set(range(len(data)))
        lista_ids_iguais = []
        print_with_time(f"Processando {len(data)} registros para itens iguais...")
        for i, item in enumerate(tqdm(data, desc="Buscando Similares")):
            if i not in keep_indices:
                continue
            
            id_tram = item["Id"]       
            reg = self.qdrant_utils.get_id(id_tram, self.collection_name)
            if reg is None:                
                print_with_time(f"Aviso: Id {id_tram} não encontrado no Qdrant, pulando")
                continue
            else:
                query_embedding = reg["Embedding"]

            if query_embedding is None:
                print_with_time(f"Aviso: Embedding do id: {id_tram} vazio, pulando")
                continue
            
            # Perform similarity search using classifica_textoBll
            try:
                result = self.classifica_texto.search_similarities(
                    collection_name=self.collection_name,
                    query_embedding=query_embedding,
                    id_a_classificar=id_tram,
                    TabelaOrigem="textos_treinamento",
                    itens_limit=self.itens_limit,
                    gravar_log=False
                )
                results = result.ListaSimilaridade or []
            except Exception as e:
                print_with_time(f"Erro ao buscar similares para Id {id_tram}: {e}")
                continue

            for similar_item in results:
                sim = similar_item.Similaridade or 0
                neighbor_id = similar_item.IdEncontrado
                neighbor_cod_classe = similar_item.CodClasse
                if neighbor_id == id_tram:
                    continue

                # Mark for removal if similarity exceeds threshold and same class
                if sim >= similarity_threshold_equal and item["CodClasse"] == neighbor_cod_classe:
                    already_exist = False
                    for idgual in lista_ids_iguais:
                        already_exist = ((idgual.id == id_tram) or (idgual.idIgual == neighbor_id) or 
                                         (idgual.idIgual == id_tram) or (idgual.id == neighbor_id))
                        if already_exist:
                            break
                    
                    if not already_exist:
                        lista_ids_iguais.append(
                            idIguaisModule.IdsIguais(id=id_tram, idIgual=neighbor_id)
                        )
          
        # Insert duplicates into database
        try:
            itens_inseridos = self.id_iguais_bll.commit_lista_ids_iguais(lista_ids_iguais)
            if itens_inseridos > 0:
                print_with_time(f"IdsIguais inseridos no banco em IdsIguais: {itens_inseridos}")
            else:
                print_with_time("Nenhum IdsIguais inserido, lista vazia ou erro na inserção.")

        except Exception as e:
            print_error(f"Erro ao inserir IdsIguais no banco: {e}")
            raise
        
        # Atualiza BuscouIgual para os itens processados para não processa-los novamente
        listaIds = (item["Id"] for item in data)
        self.id_iguais_bll.set_buscou_igual(listaIds)
        

        strRetorno = f"Processados {len(data)} textos, faltam {self._get_data_to_process(auxFilter)}, ids iguais encontrados {len(lista_ids_iguais)}."
        print_with_time(strRetorno)

        return  {
                "status": "Sucesso",
                "message": strRetorno,
                }

    def generate_ids_iguais_start(self):
        """
        Start processing the dataset for equal IDs.
        """
        print_with_time(f"Iniciando processamento de search_ids_iguais...")
        result = self._generate_ids_equal()
        print_with_time(result)
        return result

    def generate_ids_colliding_start(self):
        """
        Start processing the dataset for colliding IDs.
        """
        print_with_time(f"Iniciando processamento de generate_ids_colliding...")
        return self._genetare_ids_colliding()
        