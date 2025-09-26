import os
from fastapi import FastAPI, HTTPException,Depends
from pydantic import BaseModel
import time
from typing import Optional
from fastapi.responses import JSONResponse
import common
from fastapi import APIRouter
import bll.embeddingsBll as embeddingsModule
import bll.classifica_textoBll as classifica_textoBllModule    
from common import get_db
from sqlalchemy.orm import Session

bllEmbeddings = None

router = APIRouter()

def init():
    global bllEmbeddings    
    from main import localconfig  # importa localconfig do main.py    
    if bllEmbeddings is None:
        bllEmbeddings = embeddingsModule.Embeddings(localconfig)  # inicializa modelos (carrega embeddings)
        bllEmbeddings.load_model_and_embendings("train")  # carrega os embeddings finais

#endpoint para classificação de texto
@router.post("/classificaTexto")
async def ClassificaTexto(texto: str,
                          id_a_classificar: Optional[int] = None,
                          TabelaOrigem:Optional[str] = "",
                          db: Session = Depends(get_db)  ):    
    # Executar script de restauração
    init()  # inicializa bllEmbeddings se ainda não foi inicializado    

    try:     
        classifica_textoBll = classifica_textoBllModule.classifica_textoBll(bllEmbeddings,db)
        return classifica_textoBll.classifica_texto(texto,id_a_classificar,TabelaOrigem, top_k=20)
    
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em ClassificaTexto : {str(e)}")
    

    

