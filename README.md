# 🏦 API de Clasificación de Riesgo Crediticio

API REST para predecir la probabilidad de incumplimiento de pago de clientes.

## 📊 Modelo

- **Algoritmo:** LightGBM con Feature Engineering
- **Métrica:** ROC-AUC 76.42%
- **Técnica de balanceo:** Undersampling

## 🚀 Instalación Local

### 1. Clonar/Crear el proyecto

```bash
mkdir credit-risk-api
cd credit-risk-api
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Agregar los archivos del modelo

Asegúrate de tener estos 3 archivos en la carpeta (descargados de Colab):
- `modelo_credito.pkl`
- `scaler.pkl`
- `features.json`

### 5. Ejecutar la API

```bash
uvicorn app:app --reload
```

La API estará disponible en: http://localhost:8000

## 📖 Documentación

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🔌 Endpoints

### GET /
Información general de la API

### GET /health
Estado de salud de la API

### POST /predict
Predicción individual

**Request:**
```json
{
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
```

**Response:**
```json
{
    "probabilidad_incumplimiento": 0.2534,
    "clasificacion": "BAJO_RIESGO",
    "score_credito": 711,
    "recomendacion": "✅ APROBAR - Cliente con buen perfil, riesgo moderado-bajo",
    "confianza": "Alta"
}
```

## 🐳 Docker (Opcional)

### Construir imagen
```bash
docker build -t credit-risk-api .
```

### Ejecutar contenedor
```bash
docker run -p 8000:8000 credit-risk-api
```

## ☁️ Despliegue en Railway

### Opción 1: Desde GitHub (Recomendada)

1. **Sube tu código a GitHub**
   ```bash
   git init
   git add .
   git commit -m "API de riesgo crediticio"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/credit-risk-api.git
   git push -u origin main
   ```

2. **Ve a [railway.app](https://railway.app)** y crea una cuenta con GitHub

3. **Crear nuevo proyecto:**
   - Click en "New Project"
   - Selecciona "Deploy from GitHub repo"
   - Autoriza Railway a acceder a tus repos
   - Selecciona `credit-risk-api`

4. **Railway detectará automáticamente** la configuración y desplegará

5. **Generar dominio público:**
   - Ve a tu servicio → Settings → Networking
   - Click en "Generate Domain"
   - Tu API estará en: `https://credit-risk-api-production-xxxx.up.railway.app`

### Opción 2: Con Railway CLI

```bash
# Instalar CLI
npm install -g @railway/cli

# Login
railway login

# Inicializar proyecto
railway init

# Desplegar
railway up
```

### Variables de entorno (opcional)
Si necesitas configurar variables:
- Ve a tu servicio → Variables
- Agrega las que necesites (ej: `DEBUG=false`)

## 📝 Notas

- Los valores de `DAYS_*` son negativos (días hacia atrás)
- `EXT_SOURCE_*` van de 0 a 1 (scores de bureaus de crédito)
- El score crediticio va de 300 (malo) a 850 (excelente)

## 🧪 Probar con cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"AMT_INCOME_TOTAL":150000,"AMT_CREDIT":500000,"AMT_ANNUITY":25000,"AMT_GOODS_PRICE":450000,"DAYS_BIRTH":-12000,"DAYS_EMPLOYED":-2000,"DAYS_REGISTRATION":-5000,"DAYS_ID_PUBLISH":-3000,"EXT_SOURCE_1":0.5,"EXT_SOURCE_2":0.6,"EXT_SOURCE_3":0.55,"CNT_FAM_MEMBERS":2,"FLAG_OWN_CAR":"Y","FLAG_OWN_REALTY":"Y","REGION_RATING_CLIENT":2,"REGION_RATING_CLIENT_W_CITY":2,"DEF_30_CNT_SOCIAL_CIRCLE":0,"DEF_60_CNT_SOCIAL_CIRCLE":0,"OBS_30_CNT_SOCIAL_CIRCLE":2,"OBS_60_CNT_SOCIAL_CIRCLE":2}'
```

## 🐍 Probar con Python

```python
import requests

url = "http://localhost:8000/predict"  # Local
# url = "https://TU-APP.up.railway.app/predict"  # Railway (cambiar por tu URL)
data = {
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

response = requests.post(url, json=data)
print(response.json())
```
