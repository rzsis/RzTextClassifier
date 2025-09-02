# main.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi import APIRouter

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def raiz():
    return """
    <!doctype html>
    <html lang="pt-br">
      <head>
        <meta charset="utf-8">
        <title>Servidor OK</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          body{margin:0;display:flex;align-items:center;justify-content:center;height:100vh;font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;background:#f6f7f9}
          .card{background:#fff;padding:24px 28px;border-radius:14px;box-shadow:0 6px 24px rgba(0,0,0,.08);text-align:center}
          h1{margin:0 0 6px 0;color:#157347;font-size:28px}
          small{color:#6b7280}
        </style>
      </head>
      <body>
        <div class="card">
          <h1>Servidor OK</h1>
          <small>Aplicação: "RzTextClassifier"</small>
        </div>
      </body>
    </html>
    """