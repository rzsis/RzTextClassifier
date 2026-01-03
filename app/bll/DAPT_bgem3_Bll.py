from ast import Raise
import datetime
import os
from pathlib import Path
import re
import time
from sympy import false
from typing_extensions import runtime
import numpy as np
from requests import Session
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from common import print_with_time, print_error
from collections import Counter
from sqlalchemy import text
import gpu_utils as gpu_utilsModule
import localconfig
from bll.embeddingsBll import get_bllEmbeddings
import json


#Gera um arquivo json para fazer DAPT (Domain-Adaptive Pretraining) no bgem3 model
class DAPT_bge_m3:
    def __init__(self,  session: Session):     
        self.session = session
        self.localconfig = localconfig
        self.config = localconfig.read_config()
        self.embeddingsTrain = localconfig.getEmbeddingsTrain()
        self.model_path = Path(self.config["model_path"])        

        # Validate model directory
        if not os.path.isdir(self.model_path):            
            raise RuntimeError(f"Diretório do modelo não encontrado: {self.model_path}")
        

    #faz a consulta no banco de dados para obter os dados a serem processados
    def _fetch_data(self) -> list:   
        qtdPalavrasMinima = 80
        qtdPalavrasMaxima = 2100#faz até 2100 pois ele só salva até 2048 e deixa uma margem de segurança

        query = f"""
                (
                    select tc.TxtTreinamento,tc.QtdPalavras,min(tc.id) as id
                    from textos_classificar tc
                    where 
                    (  
                       tc.id in (select stc.idbase from sugestao_textos_classificar stc)
                    )
                    and   tc.TxtTreinamento <> '' and tc.QtdPalavras  > {qtdPalavrasMinima} and tc.QtdPalavras  < {qtdPalavrasMaxima}                
                )
                UNION 
                /*Aqui pede todos os textos que não estejam como similares que são diferentes*/
                (
                    select tc.TxtTreinamento,tc.QtdPalavras,min(tc.id) as id
                    from textos_classificar tc
                    where 
                    (
                       tc.id not in (select stc.idbase from sugestao_textos_classificar stc) or 
                       tc.id not in (select stc2.idsimilar from sugestao_textos_classificar stc2)
                    )
                    and   tc.TxtTreinamento <> '' and tc.QtdPalavras  > {qtdPalavrasMinima} and tc.QtdPalavras  < {qtdPalavrasMaxima}
                    group by tc.TxtTreinamento    
                )
                Order by QtdPalavras asc
        """

        # Busca dados do banco de dados
        try:
            result = self.session.execute(text(query)).mappings().all()
            dados = [dict(row) for row in result]
                        
            return dados
        except Exception as e:
            raise RuntimeError(f"Erro executando consulta no banco de dados: {e}")
        
    ### Aqui gerar o dataset DAPT (substituído pelo bloco abaixo)        
    def _save_dapt_file(self, dapt_data: list,comp_filename:str) -> str:
        output_path = self.localconfig.get("dataset_path")        
        os.makedirs(output_path, exist_ok=True)        
        dapt_file_path = os.path.join(output_path, f"dapt_{comp_filename}.dapt")
        try:
            with open(dapt_file_path, 'w', encoding='utf-8') as f:
                json.dump(dapt_data, f, ensure_ascii=False, indent=2)
            result = f"Dataset DAPT salvo com sucesso em: {dapt_file_path}" + "\n"
            print_with_time(result)
            print_with_time(f"Total de documentos incluídos no dapt.json: {len(dapt_data)}")
            dapt_data.clear()  # Limpa a lista após salvar o arquivo
            return result
        except Exception as e:
            print_error(f"Erro ao salvar o arquivo dapt.json: {e}")
            raise RuntimeError(f"Falha ao gravar o arquivo dapt.json: {e}")

    #Inicia o processo de geração de embeddings
    def start(self):
        iniTime = time.time()  
        print_with_time(f"Iniciando geração de arquivos JSON para DAPT de {self.model_path} : {iniTime}")
        dados = self._fetch_data()
        qtdreg = len(dados)
        if qtdreg == 0:
            return {"status": "Completo",
                    "message": f"Não há dados para gerar DataSet DAPT. "}
    
        print_with_time(f"Total de registros a processar: {len(dados)}")
        tmpErros = ""
        processados = 0        
        dapt_dataset = []
        nivel_palavras = 1
        result = ""
        #gera os dadasets baseados na quantidade de caracteres para otimizar o treinamento
        for i in tqdm(range(0, len(dados) ), desc=f"Exportando dados para DAPT"):
            ### Aqui gerar o dataset DAPT
            row = dados[i]            
            try:
                txtTreinamento = row['TxtTreinamento'].strip()
                txtTreinamento = re.sub(r'\{[A-Za-z]{1,12}\}', '', txtTreinamento)  # Remove tags {TAG}
                qtdPalavras    = row['QtdPalavras']                
                
                dapt_dataset.append({"text": txtTreinamento,
                                     "id": row["id"]
                                     })
                processados += 1
                if (nivel_palavras == 1) and (qtdPalavras > 128):
                    nivel_palavras = 2
                    result += self._save_dapt_file(dapt_dataset,"0128")
                elif (nivel_palavras == 2) and (qtdPalavras >= 256):
                    nivel_palavras = 3
                    result += self._save_dapt_file(dapt_dataset,"0256")                    
                elif (nivel_palavras == 3) and (qtdPalavras >= 512):
                    nivel_palavras = 4
                    result += self._save_dapt_file(dapt_dataset,"0512")                    
                elif (nivel_palavras == 4) and (qtdPalavras >= 768):
                    nivel_palavras = 5
                    result += self._save_dapt_file(dapt_dataset,"0768")                    
                elif (nivel_palavras == 5) and (qtdPalavras >= 1024):
                    nivel_palavras = 6
                    result += self._save_dapt_file(dapt_dataset,"1024")                    
                elif (nivel_palavras == 6) and (qtdPalavras >= 1512):
                    nivel_palavras = 7
                    result += self._save_dapt_file(dapt_dataset,"1512")                            
                elif (nivel_palavras == 7) and (qtdPalavras >= 1768):
                    nivel_palavras = 8
                    result += self._save_dapt_file(dapt_dataset,"1768")                
                elif (nivel_palavras == 8) and (qtdPalavras >= 2048):
                    nivel_palavras = 9
                    result += self._save_dapt_file(dapt_dataset,"2048")      


            except Exception as e:
                tmpErros += f"Erro ao processar registro {i+1}: {e}\n"
                print_with_time(f"Erro ao processar registro {i+1}: {e}")
                continue
            
        # Salva o ultimo arquivo DAPT com o saldo de dados
        result +=  self._save_dapt_file(dapt_dataset,"2048")                    

        elapsed     = time.time() - iniTime
        str_elapsed = f"Duração: {elapsed/60:.2f} min"
        print_with_time(f"Processamento finalizado. Total processado: {processados}. {str_elapsed}")

        if tmpErros != "":
            return {"status": "Processado com erros",
                    "message": f"Erros {tmpErros} registros. processados {processados} registros, {str_elapsed}"}
        else:
            return {"status": "Sucesso",
                    "message": f"{result} tempo decorrido:  {str_elapsed}"}