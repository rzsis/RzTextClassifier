#embeddings_Generate_Controller.py
from sqlalchemy.orm import Session
from fastapi import FastAPI, HTTPException,Depends
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from common import print_with_time, print_error, get_db, get_localconfig
from bll.embeddings_generate import GenerateEmbeddings as bllGenerateEmbeddings
from bll.search_ids_iguais import SearchIdIguais as bllSearchIdsIguais
router = APIRouter()

#faz o treinamento a montagem do banco dos embeddings
@router.post("/generateembeddings")
async def GenerateEmbeddings(
    db: Session = Depends(get_db),
    localcfg = Depends(get_localconfig),
):    
    try:        
        generator = bllGenerateEmbeddings(session=db, localcfg=localcfg)        
        generator.start()
        return {"status": "success", "message": f"Embeddings generated for split "}
    except Exception as e:        
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")
    

#endpoint para classificação de texto
@router.post("/search_ids_iguais")
async def search_ids_iguais(texto: str = "",
    db: Session = Depends(get_db),
    localcfg = Depends(get_localconfig)
    ):

    try:
        searchIdsIguais =  bllSearchIdsIguais(session=db, localcfg=localcfg)
        return  searchIdsIguais.start()
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em found_ids_iguais : {str(e)}")    

