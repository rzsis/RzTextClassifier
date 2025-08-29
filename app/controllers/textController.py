import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from typing import Optional
from fastapi.responses import JSONResponse
import main
import common
from fastapi import APIRouter

router = APIRouter()

#endpoint para listar todos os bancos de dados
@router.post("/ClassificaTexto")
async def ClassificaTexto():    
    # Executar script de restauração
    try:
        return ""
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em getdatabaselist : {str(e)}")
    

