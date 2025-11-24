import os
from fastapi import FastAPI, HTTPException,Depends
from pydantic import BaseModel
import time
from typing import Optional
from fastapi.responses import JSONResponse
import common
from fastapi import APIRouter
from common import get_session_db
from sqlalchemy.orm import Session
from bll.edit_textos_treinamentoBLL import edit_textos_treinamentoBLL as edit_textosTreinamentoBLLModule

router = APIRouter()
   

#endpoint que vai editar o texto de treinamento em textos_treinamento
@router.post("/edit_texto_treinamento")
async def edit_texto_treinamento(id: int,
                              codClasse: int,
                              codUser:int,
                              session: Session = Depends(get_session_db)  ):
    try:
        edit_textosTreinamentoBll = edit_textosTreinamentoBLLModule(session)
        return edit_textosTreinamentoBll.editar_texto_treinamento(id, codClasse)    
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em edit_texto_treinamento : {str(e)}")


