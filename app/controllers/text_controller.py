import os
from fastapi import FastAPI, HTTPException,Depends
from pydantic import BaseModel
import time
from typing import Optional
from fastapi.responses import JSONResponse
import common
from fastapi import APIRouter
import bll.embeddingsBll as embeddingsBllModule
import bll.classifica_textoBll as classifica_textoBllModule    
from common import get_session_db
from sqlalchemy.orm import Session
import bll.embeddingsBll as embeddingsBllModule
from bll.classifica_textos_pendentesBll import ClassificaTextosPendentesBll as classifica_textos_pendentesBllModule
from bll.sugere_textos_classificarBll import sugere_textos_classificarBll as sugere_textos_classificarBllModule
from bll.move_sugestao_treinamentoBLL import move_sugestao_treinamentoBLL as move_sugestao_treinamentoBllModule
from bll.indexa_textos_classificarBll import indexa_textos_classificarBll as indexa_textos_classificarBllModule
from bll.busca_textoBLL import busca_textoBLL as busca_textoBLLModule
from bll.indexa_textos_treinamentoBll import indexa_textos_treinamentoBll as indexa_textos_treinamentoBllModule

router = APIRouter()
   

#endpoint para classificação de texto
@router.post("/classificaTexto")
async def ClassificaTexto(texto: str,
                          id_a_classificar: Optional[int] = None,
                          TabelaOrigem:Optional[str] = "",
                          session: Session = Depends(get_session_db)  ):    
    # Executar script de restauração
    embeddingsBllModule.initBllEmbeddings(session=session)  # inicializa bllEmbeddings se ainda não foi inicializado  

    try:     
        classifica_textoBll = classifica_textoBllModule.classifica_textoBll(embeddingsBllModule.bllEmbeddings,session)
        return classifica_textoBll.classifica_texto(texto,
                                                    id_a_classificar,
                                                    TabelaOrigem, 
                                                    limite_itens=20, 
                                                    gravar_log=True)
    
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em ClassificaTexto : {str(e)}")
    
    
#classifica os textos pendentes em textos_classificar
@router.post("/classifica_textos_pendentes")
async def classifica_textos_pendentes(session: Session = Depends(get_session_db)  ):        
    try:     
        embeddingsBllModule.initBllEmbeddings(session=session)  # inicializa bllEmbeddings se ainda não foi inicializado          
        classifica_textos_pendentesBll = classifica_textos_pendentesBllModule(session)
        return classifica_textos_pendentesBll.classifica_textos_pendentes()
        
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em classifica_textos_pendentes : {str(e)}")
    
#endpoint para sugere textos a classificar
@router.post("/sugere_textos_classificar")
async def sugere_textos_classificar(NivelBuscaSimilar:int=0,
    session: Session = Depends(get_session_db)):        

    try:     
        embeddingsBllModule.initBllEmbeddings(session=session)  # inicializa bllEmbeddings se ainda não foi inicializado          
        sugere_textos_classificarBll = sugere_textos_classificarBllModule(session)
        return sugere_textos_classificarBll.sugere_textos_para_classificar(NivelBuscaSimilar)
        
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em sugere_textos_classificar : {str(e)}")    
    
#endpoint para indexar textos a classificar
@router.post("/indexa_textos_classificar")
async def indexa_textos_classificar(session: Session = Depends(get_session_db)  ):        
    try:     
        embeddingsBllModule.initBllEmbeddings(session=session)  # inicializa bllEmbeddings se ainda não foi inicializado          
        indexa_textos_classificar = indexa_textos_classificarBllModule(session=session)
        return indexa_textos_classificar.indexa_textos_classificar()
        
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em indexa_textos_classificar : {str(e)}")      

#endpoint para indexar textos a treinamento
@router.post("/indexa_textos_treinamento")
async def indexa_textos_treinamento(session: Session = Depends(get_session_db)  ):        
    try:     
        embeddingsBllModule.initBllEmbeddings(session=session)  # inicializa bllEmbeddings se ainda não foi inicializado          
        indexa_textos_treinamento = indexa_textos_treinamentoBllModule(session=session)
        return indexa_textos_treinamento.indexa_textos_treinamento()
        
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em indexa_textos_treinamento : {str(e)}")             
    

#endpoint para mover sugestão de treinamento
@router.post("/move_sugestao_treinamento")
async def move_sugestao_treinamento(idbase:int,
                                    idsimilar:int,
                                    codclasse:int,                                    
                                    coduser:int,
                                    mover_com_colidencia:bool=False,
                                    session: Session = Depends(get_session_db)  ):        

    try:             
        move_sugestao_treinamentoBll = move_sugestao_treinamentoBllModule(session)

        return move_sugestao_treinamentoBll.move_sugestao_treinamento(idBase=idbase,
                                                                idSimilar=idsimilar,
                                                                codClasse=codclasse,
                                                                coduser=coduser,
                                                                mover_com_colidencia=mover_com_colidencia)
        
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em move_sugestao_treinamento : {str(e)}")    
    

#endpoint para buscar textos treinamento
@router.post("/busca_textos")
async def busca_textos( id:int,#passar 0 se não quiser filtrar por id
                        codclasse:int,#passar 0 se não quiser filtrar por codclasse
                        texto:str,#obrigatório
                        data_inicial:str ,#passar "" se não quiser filtrar por data_inicial
                        data_final:str ,#passar "" se não quiser filtrar por data_final
                        similaridade_minima:float,#passar 0 se não quiser filtrar por similaridade mínima valor deve estar entre 0 e 1
                        collection_name:str,#obrigatório final ou train define se vai buscar na coleção ja treinada ou de textos a classificar
                        session: Session = Depends(get_session_db)):        

    try:             
        busca_textoBLL = busca_textoBLLModule(embeddingsBllModule.get_bllEmbeddings(session=session),session=session)
        return busca_textoBLL.busca_texto(id=id,
                                          codclasse=codclasse,
                                          texto=texto,
                                          data_inicial=data_inicial,
                                          data_final=data_final,
                                          similaridade_minima=similaridade_minima,
                                          collection_name=collection_name)
        
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em busca_textos : {str(e)}")    