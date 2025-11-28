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
from bll.finetuning_bgem3_Bll import finetuning_bge_m3 as finetuning_bge_m3Module

router = APIRouter()
   

#endpoint que vai editar o texto de treinamento em textos_treinamento
@router.post("/bge_m3_finetunning")
async def bge_m3_finetunning(session = Depends(get_session_db)  ):
    try:
        finetuning_bge_m3Bll = finetuning_bge_m3Module(session)
        return finetuning_bge_m3Bll.start()
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em bge_m3_finetunning : {str(e)}")


