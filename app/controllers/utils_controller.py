import os
from fastapi import FastAPI, HTTPException,Depends
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
import common
from fastapi import APIRouter
from common import get_session_db
from sqlalchemy.orm import Session
import dbClasses.classes_utils
from bll.exportLLM.export_LLM import Export_LLM


router = APIRouter()
   

#atualiza as classes
@router.post("/update_classes")
async def update_classes(session: Session = Depends(get_session_db)  ):
    try:
       dbClasses.classes_utils.classes_utils_singleton.atualizar_lista_classes()
       return JSONResponse(content={"message": "Lista de classes atualizada com sucesso."})
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em update_classes : {str(e)}")

#Exporta os textos para o dataset LLM
@router.post("/export_llm")
async def export_llm(session: Session = Depends(get_session_db)):
    try:
        exporter_llm = Export_LLM(session)
        result = exporter_llm.start()
        return JSONResponse(content=result)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro ao exportar dataset LLM: {str(e)}")
