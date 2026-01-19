"""
Script para probar la API localmente
=====================================
Ejecutar: python test_api.py
"""

import requests
import json

# URL de la API (local)
BASE_URL = "http://localhost:8000"

def test_health():
    """Probar endpoint de salud"""
    print("=" * 50)
    print("🔍 Probando /health")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_predict_bajo_riesgo():
    """Probar predicción con cliente de bajo riesgo"""
    print("\n" + "=" * 50)
    print("🟢 Probando /predict - Cliente BAJO RIESGO")
    print("=" * 50)
    
    # Cliente con buen perfil
    cliente = {
        "AMT_INCOME_TOTAL": 300000,      # Buen ingreso
        "AMT_CREDIT": 400000,            # Crédito moderado
        "AMT_ANNUITY": 20000,
        "AMT_GOODS_PRICE": 380000,
        "DAYS_BIRTH": -15000,            # ~41 años
        "DAYS_EMPLOYED": -3650,          # ~10 años empleado
        "DAYS_REGISTRATION": -8000,
        "DAYS_ID_PUBLISH": -4000,
        "EXT_SOURCE_1": 0.7,             # Buenos scores externos
        "EXT_SOURCE_2": 0.75,
        "EXT_SOURCE_3": 0.68,
        "CNT_FAM_MEMBERS": 3,
        "FLAG_OWN_CAR": "Y",             # Tiene carro
        "FLAG_OWN_REALTY": "Y",          # Tiene propiedad
        "REGION_RATING_CLIENT": 1,       # Buena región
        "REGION_RATING_CLIENT_W_CITY": 1,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0,   # Sin defaults en círculo social
        "DEF_60_CNT_SOCIAL_CIRCLE": 0,
        "OBS_30_CNT_SOCIAL_CIRCLE": 5,
        "OBS_60_CNT_SOCIAL_CIRCLE": 5
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=cliente)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_predict_alto_riesgo():
    """Probar predicción con cliente de alto riesgo"""
    print("\n" + "=" * 50)
    print("🔴 Probando /predict - Cliente ALTO RIESGO")
    print("=" * 50)
    
    # Cliente con perfil riesgoso
    cliente = {
        "AMT_INCOME_TOTAL": 50000,       # Bajo ingreso
        "AMT_CREDIT": 800000,            # Crédito muy alto
        "AMT_ANNUITY": 50000,            # Alta anualidad
        "AMT_GOODS_PRICE": 750000,
        "DAYS_BIRTH": -8000,             # ~22 años (joven)
        "DAYS_EMPLOYED": -180,           # Solo 6 meses empleado
        "DAYS_REGISTRATION": -500,
        "DAYS_ID_PUBLISH": -200,
        "EXT_SOURCE_1": 0.2,             # Malos scores externos
        "EXT_SOURCE_2": 0.25,
        "EXT_SOURCE_3": 0.18,
        "CNT_FAM_MEMBERS": 5,            # Familia grande
        "FLAG_OWN_CAR": "N",             # Sin carro
        "FLAG_OWN_REALTY": "N",          # Sin propiedad
        "REGION_RATING_CLIENT": 3,       # Mala región
        "REGION_RATING_CLIENT_W_CITY": 3,
        "DEF_30_CNT_SOCIAL_CIRCLE": 3,   # Defaults en círculo social
        "DEF_60_CNT_SOCIAL_CIRCLE": 2,
        "OBS_30_CNT_SOCIAL_CIRCLE": 10,
        "OBS_60_CNT_SOCIAL_CIRCLE": 8
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=cliente)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_predict_batch():
    """Probar predicción en lote"""
    print("\n" + "=" * 50)
    print("📦 Probando /predict_batch - Múltiples clientes")
    print("=" * 50)
    
    clientes = [
        {
            "AMT_INCOME_TOTAL": 200000,
            "AMT_CREDIT": 300000,
            "AMT_ANNUITY": 15000,
            "AMT_GOODS_PRICE": 280000,
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -2000,
            "DAYS_REGISTRATION": -5000,
            "DAYS_ID_PUBLISH": -3000,
            "EXT_SOURCE_1": 0.6,
            "EXT_SOURCE_2": 0.65,
            "EXT_SOURCE_3": 0.58,
            "CNT_FAM_MEMBERS": 2,
            "FLAG_OWN_CAR": "Y",
            "FLAG_OWN_REALTY": "Y",
            "REGION_RATING_CLIENT": 2,
            "REGION_RATING_CLIENT_W_CITY": 2,
            "DEF_30_CNT_SOCIAL_CIRCLE": 0,
            "DEF_60_CNT_SOCIAL_CIRCLE": 0,
            "OBS_30_CNT_SOCIAL_CIRCLE": 3,
            "OBS_60_CNT_SOCIAL_CIRCLE": 3
        },
        {
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 600000,
            "AMT_ANNUITY": 35000,
            "AMT_GOODS_PRICE": 580000,
            "DAYS_BIRTH": -9000,
            "DAYS_EMPLOYED": -500,
            "DAYS_REGISTRATION": -2000,
            "DAYS_ID_PUBLISH": -1000,
            "EXT_SOURCE_1": 0.3,
            "EXT_SOURCE_2": 0.35,
            "EXT_SOURCE_3": 0.28,
            "CNT_FAM_MEMBERS": 4,
            "FLAG_OWN_CAR": "N",
            "FLAG_OWN_REALTY": "N",
            "REGION_RATING_CLIENT": 3,
            "REGION_RATING_CLIENT_W_CITY": 3,
            "DEF_30_CNT_SOCIAL_CIRCLE": 1,
            "DEF_60_CNT_SOCIAL_CIRCLE": 1,
            "OBS_30_CNT_SOCIAL_CIRCLE": 5,
            "OBS_60_CNT_SOCIAL_CIRCLE": 4
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict_batch", json=clientes)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


if __name__ == "__main__":
    print("\n" + "🚀 INICIANDO PRUEBAS DE LA API" + "\n")
    
    resultados = {
        "health": test_health(),
        "bajo_riesgo": test_predict_bajo_riesgo(),
        "alto_riesgo": test_predict_alto_riesgo(),
        "batch": test_predict_batch()
    }
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    for test, passed in resultados.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test}: {status}")
    
    total_passed = sum(resultados.values())
    print(f"\n   Total: {total_passed}/{len(resultados)} pruebas exitosas")
