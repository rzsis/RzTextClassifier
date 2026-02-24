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


router = APIRouter()
   

#atualiza as classes
@router.post("/update_classes")
async def update_classes(session: Session = Depends(get_session_db)  ):
    try:
       dbClasses.classes_utils.classes_utils_singleton.atualizar_lista_classes()
       return JSONResponse(content={"message": "Lista de classes atualizada com sucesso."})
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em update_classes : {str(e)}")


