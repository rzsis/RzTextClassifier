import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from typing import Optional
from fastapi.responses import JSONResponse
import common
from fastapi import APIRouter

router = APIRouter()

#endpoint para classificação de texto
@router.post("/classificaTexto")
async def ClassificaTexto(strTexto: str):    
    # Executar script de restauração
    from main import embeddings

    try:
        return embeddings.classifica_texto(strTexto)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em ClassificaTexto : {str(e)}")
    

