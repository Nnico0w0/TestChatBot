#!/bin/bash

# Script de entrenamiento para TestChatBot
# Uso: bash train.sh [opciones]

echo "======================================"
echo "TestChatBot - Entrenamiento"
echo "======================================"

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo "Activando entorno virtual..."
    source venv/bin/activate
fi

# Verificar que Python está disponible
if ! command -v python &> /dev/null; then
    echo "Error: Python no está instalado"
    exit 1
fi

# Verificar dependencias
echo "Verificando dependencias..."
python -c "import torch; import nltk; import yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Faltan dependencias. Ejecuta: pip install -r requirements.txt"
    exit 1
fi

# Preprocesar datos si no existen
if [ ! -f "data/processed/train.pkl" ]; then
    echo "Preprocesando datos..."
    python src/preprocessing.py
    if [ $? -ne 0 ]; then
        echo "Error en el preprocesamiento"
        exit 1
    fi
fi

# Iniciar entrenamiento
echo "Iniciando entrenamiento..."
python src/train.py "$@"

echo "======================================"
echo "Entrenamiento finalizado"
echo "======================================"
