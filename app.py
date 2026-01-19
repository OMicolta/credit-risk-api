"""
API de Clasificación de Riesgo Crediticio
=========================================
Modelo: LightGBM con Feature Engineering
Métrica: ROC-AUC 76.42%

Uso:
    uvicorn app:app --reload
    
Endpoints:
    GET  /              → Info de la API
    GET  /health        → Estado de salud
    POST /predict       → Predicción individual
    POST /predict_batch → Predicciones múltiples
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import json
import numpy as np
import pandas as pd
from feature_engineering import apply_feature_engineering

# ============================================
# CARGAR ARTEFACTOS DEL MODELO
# ============================================
import os
import sys
import traceback

model = None
scaler = None
feature_names = None
load_error = None

try:
    # Obtener directorio del script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, 'modelo_credito.pkl')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    features_path = os.path.join(BASE_DIR, 'features.json')

    print(f"📁 Directorio base: {BASE_DIR}", file=sys.stderr)
    print(f"📁 Archivos en directorio: {os.listdir(BASE_DIR)}", file=sys.stderr)

    model = joblib.load(model_path)
    print("✅ Modelo cargado", file=sys.stderr)

    scaler = joblib.load(scaler_path)
    print("✅ Scaler cargado", file=sys.stderr)

    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    print(f"✅ Features cargadas: {len(feature_names)}", file=sys.stderr)

except Exception as e:
    load_error = f"{type(e).__name__}: {str(e)}"
    print(f"❌ Error cargando artefactos: {load_error}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)

# ============================================
# CONFIGURACIÓN DE LA API
# ============================================
app = FastAPI(
    title="API de Riesgo Crediticio",
    description="Predice la probabilidad de incumplimiento de pago de un cliente",
    version="1.0.0",
    docs_url="/docs",  # Documentación Swagger
    redoc_url="/redoc"  # Documentación ReDoc
)

# Permitir CORS (para consumir desde frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# ESQUEMAS DE DATOS (Pydantic)
# ============================================
class ClienteInput(BaseModel):
    """Datos de entrada del cliente para predicción"""
    
    # Información del préstamo
    AMT_INCOME_TOTAL: float = Field(..., description="Ingreso total del cliente", example=150000)
    AMT_CREDIT: float = Field(..., description="Monto del crédito solicitado", example=500000)
    AMT_ANNUITY: float = Field(..., description="Anualidad del préstamo", example=25000)
    AMT_GOODS_PRICE: float = Field(..., description="Precio del bien a comprar", example=450000)
    
    # Información personal
    DAYS_BIRTH: int = Field(..., description="Días desde nacimiento (negativo)", example=-12000)
    DAYS_EMPLOYED: int = Field(..., description="Días empleado (negativo)", example=-2000)
    DAYS_REGISTRATION: int = Field(..., description="Días desde registro", example=-5000)
    DAYS_ID_PUBLISH: int = Field(..., description="Días desde publicación de ID", example=-3000)
    
    # Scores externos (muy importantes)
    EXT_SOURCE_1: Optional[float] = Field(0.5, description="Score externo 1", example=0.5)
    EXT_SOURCE_2: Optional[float] = Field(0.5, description="Score externo 2", example=0.6)
    EXT_SOURCE_3: Optional[float] = Field(0.5, description="Score externo 3", example=0.55)
    
    # Información familiar
    CNT_FAM_MEMBERS: int = Field(2, description="Número de miembros de familia", example=2)
    
    # Flags
    FLAG_OWN_CAR: str = Field("N", description="Tiene carro (Y/N)", example="Y")
    FLAG_OWN_REALTY: str = Field("Y", description="Tiene propiedad (Y/N)", example="Y")
    
    # Región
    REGION_RATING_CLIENT: int = Field(2, description="Rating de la región (1-3)", example=2)
    REGION_RATING_CLIENT_W_CITY: int = Field(2, description="Rating región con ciudad", example=2)
    
    # Círculo social
    DEF_30_CNT_SOCIAL_CIRCLE: float = Field(0, description="Defaults 30 días círculo social", example=0)
    DEF_60_CNT_SOCIAL_CIRCLE: float = Field(0, description="Defaults 60 días círculo social", example=0)
    OBS_30_CNT_SOCIAL_CIRCLE: float = Field(0, description="Observaciones 30 días", example=2)
    OBS_60_CNT_SOCIAL_CIRCLE: float = Field(0, description="Observaciones 60 días", example=2)

    class Config:
        schema_extra = {
            "example": {
                "AMT_INCOME_TOTAL": 150000,
                "AMT_CREDIT": 500000,
                "AMT_ANNUITY": 25000,
                "AMT_GOODS_PRICE": 450000,
                "DAYS_BIRTH": -12000,
                "DAYS_EMPLOYED": -2000,
                "DAYS_REGISTRATION": -5000,
                "DAYS_ID_PUBLISH": -3000,
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": 0.6,
                "EXT_SOURCE_3": 0.55,
                "CNT_FAM_MEMBERS": 2,
                "FLAG_OWN_CAR": "Y",
                "FLAG_OWN_REALTY": "Y",
                "REGION_RATING_CLIENT": 2,
                "REGION_RATING_CLIENT_W_CITY": 2,
                "DEF_30_CNT_SOCIAL_CIRCLE": 0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 0,
                "OBS_30_CNT_SOCIAL_CIRCLE": 2,
                "OBS_60_CNT_SOCIAL_CIRCLE": 2
            }
        }


class PrediccionOutput(BaseModel):
    """Resultado de la predicción"""
    probabilidad_incumplimiento: float = Field(..., description="Probabilidad de no pagar (0-1)")
    clasificacion: str = Field(..., description="ALTO_RIESGO o BAJO_RIESGO")
    score_credito: int = Field(..., description="Score crediticio (300-850)")
    recomendacion: str = Field(..., description="Recomendación para el analista")
    confianza: str = Field(..., description="Nivel de confianza de la predicción")


# ============================================
# FUNCIONES AUXILIARES
# ============================================
def preparar_datos(cliente: ClienteInput) -> pd.DataFrame:
    """Convierte input del cliente a DataFrame con Feature Engineering"""
    
    # Crear DataFrame con los datos de entrada
    data = cliente.dict()
    df = pd.DataFrame([data])
    
    # Aplicar Feature Engineering
    df = apply_feature_engineering(df)
    
    return df


def calcular_score_credito(probabilidad: float) -> int:
    """Convierte probabilidad a score crediticio (300-850)"""
    # Invertir: menor probabilidad de incumplimiento = mayor score
    score = int(850 - (probabilidad * 550))
    return max(300, min(850, score))


def obtener_recomendacion(probabilidad: float) -> tuple:
    """Genera recomendación basada en la probabilidad"""
    if probabilidad < 0.15:
        return "BAJO_RIESGO", "✅ APROBAR - Cliente con excelente perfil crediticio", "Alta"
    elif probabilidad < 0.30:
        return "BAJO_RIESGO", "✅ APROBAR - Cliente con buen perfil, riesgo moderado-bajo", "Alta"
    elif probabilidad < 0.50:
        return "MEDIO_RIESGO", "⚠️ REVISAR - Requiere análisis adicional del analista", "Media"
    elif probabilidad < 0.70:
        return "ALTO_RIESGO", "⚠️ PRECAUCIÓN - Alto riesgo, considerar garantías adicionales", "Media"
    else:
        return "ALTO_RIESGO", "❌ RECHAZAR - Muy alto riesgo de incumplimiento", "Alta"


# ============================================
# ENDPOINTS
# ============================================
@app.get("/", tags=["Info"])
async def root():
    """Información general de la API"""
    return {
        "nombre": "API de Riesgo Crediticio",
        "version": "1.0.0",
        "modelo": "LightGBM con Feature Engineering",
        "metrica": "ROC-AUC 76.42%",
        "documentacion": "/docs",
        "estado": "activo" if model is not None else "error"
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Verificar estado de salud de la API"""
    if model is None:
        raise HTTPException(status_code=503, detail=f"Modelo no cargado: {load_error}")
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_loaded": scaler is not None,
        "features_count": len(feature_names) if feature_names else 0
    }


@app.post("/predict", response_model=PrediccionOutput, tags=["Predicciones"])
async def predecir(cliente: ClienteInput):
    """
    Realizar predicción de riesgo crediticio para un cliente.
    
    Retorna:
    - Probabilidad de incumplimiento (0-1)
    - Clasificación (ALTO_RIESGO / BAJO_RIESGO)
    - Score crediticio (300-850)
    - Recomendación para el analista
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Preparar datos
        df = preparar_datos(cliente)
        
        # Asegurar que tenemos todas las features necesarias
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Ordenar columnas según el modelo espera
        df = df[feature_names]
        
        # Escalar
        df_scaled = scaler.transform(df)
        
        # Predecir
        probabilidad = model.predict_proba(df_scaled)[0][1]
        
        # Generar resultados
        clasificacion, recomendacion, confianza = obtener_recomendacion(probabilidad)
        score = calcular_score_credito(probabilidad)
        
        return PrediccionOutput(
            probabilidad_incumplimiento=round(probabilidad, 4),
            clasificacion=clasificacion,
            score_credito=score,
            recomendacion=recomendacion,
            confianza=confianza
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/predict_batch", tags=["Predicciones"])
async def predecir_batch(clientes: List[ClienteInput]):
    """
    Realizar predicciones para múltiples clientes.
    Máximo 100 clientes por request.
    """
    if len(clientes) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 clientes por request")
    
    resultados = []
    for i, cliente in enumerate(clientes):
        try:
            resultado = await predecir(cliente)
            resultados.append({
                "index": i,
                "status": "success",
                "resultado": resultado.dict()
            })
        except Exception as e:
            resultados.append({
                "index": i,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "total": len(clientes),
        "exitosos": sum(1 for r in resultados if r["status"] == "success"),
        "resultados": resultados
    }


# ============================================
# EJECUTAR (para desarrollo local)
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
