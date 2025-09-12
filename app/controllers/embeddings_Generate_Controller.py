#embeddings_Generate_Controller.py
from sqlalchemy.orm import Session
from fastapi import FastAPI, HTTPException,Depends
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from common import print_with_time, print_error, get_db, get_localconfig
from bll.embeddings_generate import GenerateEmbeddingsModule as bllGenerateEmbeddings
from bll.generate_ids_equal_colliding import GenerateIdsIguaisCollindgs as GenerateIdsIguaisCollindgsModule
router = APIRouter()

#faz o treinamento a montagem do banco dos embeddings
@router.post("/generateembeddings")
async def GenerateEmbeddings(split: str,
    db: Session = Depends(get_db),
    localcfg = Depends(get_localconfig),
):        
    try:        
        if split not in ["train", "final"]:
            raise HTTPException(status_code=400, detail="split deve ser 'train' ou 'final'")    
        
        generator = bllGenerateEmbeddings(split, session=db, localcfg=localcfg)        
        generator.start()
        return {"status": "success", "message": f"Embeddings generated for split "}
    except Exception as e:        
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")
    

#endpoint para gerar ids iguais
@router.post("/generate_ids_iguais")
async def generate_ids_iguais(
    db: Session = Depends(get_db),
    localcfg = Depends(get_localconfig)
    ):

    try:
        GenerateIdsIguaisCollindgs =  GenerateIdsIguaisCollindgsModule(session=db, localcfg=localcfg)
        return  GenerateIdsIguaisCollindgs.generate_ids_iguais_start()
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em generate_ids_iguais : {str(e)}")    

#endpoint para gerar ids colidentes
@router.post("/generate_ids_colidentes")
async def generate_ids_colidentes(
    db: Session = Depends(get_db),
    localcfg = Depends(get_localconfig)
    ):

    try:
        GenerateIdsIguaisCollindgs =  GenerateIdsIguaisCollindgsModule(session=db, localcfg=localcfg)
        return  GenerateIdsIguaisCollindgs.generate_ids_colliding_start()
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Erro em generate_ids_iguais : {str(e)}")    

