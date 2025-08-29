import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import endpoints

# Carregar variáveis de ambiente
load_dotenv('sqlserverrestore.env')
fastApi = FastAPI()


# Dados do .env
DB_SERVER = os.getenv("DATABASE_SERVER")
DB_USERNAME = os.getenv("DATABASE_USERNAME")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD")
DB_PORT = os.getenv("DATABASE_PORT")
HTTP_PORT = int(os.getenv("HTTP_PORT", 8000))
ENDPOINT_PASSWORD = os.getenv("ENDPOINT_PASSWORD")

    
# Iniciar o servidor FastAPI
if __name__ == "__main__":
    import uvicorn
    fastApi.include_router(endpoints.router)
    uvicorn.run(fastApi, host="0.0.0.0", port=HTTP_PORT)
