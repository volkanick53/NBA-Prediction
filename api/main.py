"""
api/main.py
============
FastAPI uygulaması — giriş noktası.

Çalıştırma:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import games, projections

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("api.main")

# ---------------------------------------------------------------
# Lifespan — başlangıç / kapanış olayları
# ---------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("NBA Tahmin API başlatılıyor...")
    # İleride: model cache burada ön yüklenebilir
    yield
    log.info("NBA Tahmin API kapatılıyor.")

# ---------------------------------------------------------------
# FastAPI Uygulaması
# ---------------------------------------------------------------
app = FastAPI(
    title="🏀 NBA Prediction API",
    description=(
        "Günlük NBA maç ve oyuncu prop tahminleri. "
        "XGBoost tabanlı iki aşamalı ML pipeline: "
        "Stage 1 (Game Pace + Team Score) → Stage 2 (Player Props)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------
# CORS — Next.js frontend erişimi için
# ---------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Router'ları Kaydet
# ---------------------------------------------------------------
app.include_router(games.router,       prefix="/api", tags=["Games"])
app.include_router(projections.router, prefix="/api", tags=["Projections"])


# ---------------------------------------------------------------
# Sağlık Kontrolü
# ---------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "nba-prediction-api"}
