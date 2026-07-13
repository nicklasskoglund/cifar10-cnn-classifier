"""
Startpunkt för FastAPI-applikationen för CIFAR-10 CNN-klassificeraren.

Exponerar prediction-API:et och konfigurerar CORS så att React (Vite)-
frontenden kan anropa det både under lokal utveckling och under själva demot.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import predict

app = FastAPI(
    title="CIFAR-10 CNN Classifier API",
    description="Serverar prediktioner från en tränad CNN-modell för CIFAR-10-datasetet.",
    version="1.0.0",
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
    """Enkel health check-endpoint."""
    return {"status": "ok", "message": "CIFAR-10 CNN Classifier API körs"}