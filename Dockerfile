# Imagen base de Python
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Instalar libgomp (requerido por LightGBM)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Puerto que expone la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD gunicorn app:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
