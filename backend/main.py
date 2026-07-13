"""
Startpunkt för FastAPI-applikationen för CIFAR-10 CNN-klassificeraren.

Exponerar prediction-API:et, konfigurerar CORS så att React (Vite)-frontenden
kan anropa det, och laddar den tränade modellen redan vid serverns uppstart
(istället för vid första requesten) så att demot känns snabbt och responsivt.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Hanterar kod som ska köras vid uppstart och avstängning av servern.

    Vid uppstart: laddar CNN-modellen in i minnet en gång, så att den
    är redo direkt när första /predict-anropet kommer in.
    """
    print("Laddar CNN-modell...")
    predict.load_model()
    print("Modellen är laddad och redo.")

    yield  # servern körs här

    # (Ingen städning behövs vid avstängning just nu)


app = FastAPI(
    title="CIFAR-10 CNN Classifier API",
    description="Serverar prediktioner från en tränad CNN-modell för CIFAR-10-datasetet.",
    version="1.0.0",
    lifespan=lifespan,
)

# Tillåt Vite dev-servern (och senare den driftsatta frontenden) att anropa detta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)


@app.get("/")
async def root():
    """
    Health check-endpoint.

    Bekräftar att servern körs OCH att modellen faktiskt är laddad i minnet —
    praktiskt att snabbt kunna kolla under presentationen att allt är klart
    innan demot drar igång.
    """
    return {
        "status": "ok",
        "message": "CIFAR-10 CNN Classifier API körs",
        "model_loaded": predict.is_model_loaded(),
    }