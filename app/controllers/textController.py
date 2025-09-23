import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from typing import Optional
from fastapi.responses import JSONResponse
import common
from fastapi import APIRouter
import bll.embeddings as embeddingsModule
import bll.classifica_textoBll as classifica_textoBllModule    

bllEmbeddings = None

router = APIRouter()

def init():
    global bllEmbeddings    
    from main import localconfig  # importa localconfig do main.py    
    if bllEmbeddings is None:
        bllEmbeddings = embeddingsModule.EmbeddingsModule(localconfig)  # inicializa modelos (carrega embeddings)
        bllEmbeddings.load_model_and_embendings("train")  # carrega os embeddings finais

#endpoint para classificação de texto
@router.post("/classificaTexto")
async def ClassificaTexto(texto: str):    
    # Executar script de restauração
    init()  # inicializa bllEmbeddings se ainda não foi inicializado    

    try:     
        return classifica_textoBllModule.classifica_textoBll(bllEmbeddings).classifica_texto(texto, top_k=20)
    
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em ClassificaTexto : {str(e)}")
    

    

