#!/bin/bash

# Script de inicio rápido para TestChatBot con Docker
# Quick start script for TestChatBot with Docker

echo "======================================"
echo "TestChatBot - Docker Quick Start"
echo "======================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker no está instalado"
    echo "   Instala Docker desde: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker no está en ejecución"
    echo "   Inicia Docker y vuelve a intentarlo"
    exit 1
fi

echo "✅ Docker detectado y en ejecución"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creando archivo .env desde .env.example..."
    cp .env.example .env
    echo "✅ Archivo .env creado"
else
    echo "✅ Archivo .env ya existe"
fi
echo ""

# Create necessary directories
echo "📁 Creando directorios necesarios..."
mkdir -p data/raw data/processed models/checkpoints models/tokenizer models/final
echo "✅ Directorios creados"
echo ""

# Build and start containers
echo "🐳 Construyendo y ejecutando contenedores..."
echo "   Esto puede tardar unos minutos la primera vez..."
echo ""

docker compose up --build -d

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✅ ¡TestChatBot iniciado exitosamente!"
    echo "======================================"
    echo ""
    echo "🌐 Aplicación disponible en: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
    echo "❤️  Health Check: http://localhost:8000/health"
    echo ""
    echo "📋 Comandos útiles:"
    echo "   Ver logs:     docker compose logs -f"
    echo "   Detener:      docker compose down"
    echo "   Reiniciar:    docker compose restart"
    echo ""
    echo "⚠️  Nota: El chatbot necesita un modelo entrenado para responder."
    echo "   Ver DOCKER.md para instrucciones de entrenamiento."
    echo ""
else
    echo ""
    echo "❌ Error al iniciar los contenedores"
    echo "   Revisa los logs con: docker compose logs"
    exit 1
fi
