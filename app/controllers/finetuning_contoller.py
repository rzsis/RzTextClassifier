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
from bll.DAPT_bgem3_Bll import DAPT_bge_m3 as DAPT_bge_m3Module

router = APIRouter()
   
#Endpoint para gerar um JSON que vai ser usado para treinarmento do BGE-M3 em DAPT domain adaptation pre-training
@router.post("/dapt_bgem3")
async def dapt_bgem3(session = Depends(get_session_db)  ):
    try:
        DAPT_bge_m3Bll = DAPT_bge_m3Module(session)
        return DAPT_bge_m3Bll.start()
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em dapt_bgem3 : {str(e)}")


