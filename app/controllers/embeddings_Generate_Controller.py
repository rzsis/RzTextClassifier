#embeddings_Generate_Controller.py
from sqlalchemy.orm import Session
from fastapi import FastAPI, HTTPException,Depends
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from common import print_with_time, print_error, get_db, get_localconfig
router = APIRouter()

#faz o treinamento a montagem do banco dos embeddings
@router.post("/generateembeddings")
async def GenerateEmbeddings(
    db: Session = Depends(get_db),
    localcfg = Depends(get_localconfig),
):    
    try:        
        generator = GenerateEmbeddings(db=db, localcfg=localcfg)        
        generator.start()
        return {"status": "success", "message": f"Embeddings generated for split "}
    except Exception as e:        
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")
    

