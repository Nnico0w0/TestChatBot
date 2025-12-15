# 🐳 Docker Setup - TestChatBot

Guía para ejecutar TestChatBot usando Docker y Docker Compose.

## 📋 Requisitos Previos

- Docker instalado (versión 20.10 o superior)
- Docker Compose instalado (versión 1.29 o superior)
- Al menos 4GB de RAM disponible
- Espacio en disco: ~5GB para la imagen y modelos

## 🚀 Inicio Rápido con Docker Compose

### 1. Construir y Ejecutar

```bash
# Clonar el repositorio
git clone https://github.com/Nnico0w0/TestChatBot.git
cd TestChatBot

# Construir y ejecutar con docker-compose
docker-compose up --build
```

La aplicación estará disponible en: http://localhost:8000

### 2. Ejecutar en Segundo Plano

```bash
# Ejecutar en modo detached
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener
docker-compose down
```

### 3. Verificar Estado

```bash
# Ver estado de los contenedores
docker-compose ps

# Ver logs en tiempo real
docker-compose logs -f chatbot

# Health check
curl http://localhost:8000/health
```

## 🔧 Comandos Útiles

### Gestión del Contenedor

```bash
# Iniciar servicios
docker-compose start

# Detener servicios (sin eliminar contenedores)
docker-compose stop

# Reiniciar servicios
docker-compose restart

# Detener y eliminar contenedores
docker-compose down

# Eliminar contenedores, redes y volúmenes
docker-compose down -v
```

### Acceso al Contenedor

```bash
# Ejecutar comandos dentro del contenedor
docker-compose exec chatbot bash

# Ver logs
docker-compose logs chatbot

# Seguir logs en tiempo real
docker-compose logs -f chatbot
```

### Entrenamiento del Modelo

Si necesitas entrenar el modelo dentro del contenedor:

```bash
# Acceder al contenedor
docker-compose exec chatbot bash

# Dentro del contenedor, ejecutar entrenamiento
python src/preprocessing.py
python src/tokenizer.py
python src/train.py --epochs 10
```

O ejecutar directamente sin entrar al contenedor:

```bash
docker-compose exec chatbot python src/preprocessing.py
docker-compose exec chatbot python src/tokenizer.py
docker-compose exec chatbot python src/train.py --epochs 10
```

## 📁 Persistencia de Datos

Los siguientes directorios están montados como volúmenes para persistir datos:

- `./data` → `/app/data` - Datasets y datos procesados
- `./models` → `/app/models` - Modelos entrenados y checkpoints
- `./config.yaml` → `/app/config.yaml` - Configuración

Los datos persisten incluso si el contenedor se elimina.

## 🔍 Endpoints Disponibles

Una vez que el contenedor esté ejecutándose:

- **Interfaz Web**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Chat Endpoint**: http://localhost:8000/chat (POST)

### Ejemplo de Uso de la API

```bash
# Health check
curl http://localhost:8000/health

# Hacer una pregunta
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuándo comienzan las inscripciones?",
    "max_length": 50
  }'
```

## 🐛 Solución de Problemas

### El contenedor no inicia

```bash
# Ver logs detallados
docker-compose logs chatbot

# Reconstruir la imagen
docker-compose build --no-cache
docker-compose up
```

### Puerto 8000 en uso

Modificar `docker-compose.yml` para usar otro puerto:

```yaml
ports:
  - "8080:8000"  # Usar puerto 8080 en el host
```

### Modelo no encontrado

El chatbot necesita un modelo entrenado. Si no tienes uno:

```bash
# Entrenar el modelo dentro del contenedor
docker-compose exec chatbot bash -c "
  python src/preprocessing.py &&
  python src/tokenizer.py &&
  python src/train.py --epochs 10
"
```

### Problemas de memoria

Si el contenedor se queda sin memoria durante el entrenamiento:

```bash
# Aumentar límite de memoria en docker-compose.yml
services:
  chatbot:
    deploy:
      resources:
        limits:
          memory: 8G
```

## 🏗️ Construcción Manual (sin docker-compose)

Si prefieres usar Docker directamente:

```bash
# Construir imagen
docker build -t testchatbot:latest .

# Ejecutar contenedor
docker run -d \
  --name testchatbot \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config.yaml:/app/config.yaml \
  testchatbot:latest

# Ver logs
docker logs -f testchatbot

# Detener y eliminar
docker stop testchatbot
docker rm testchatbot
```

## 🔐 Consideraciones de Seguridad

- El contenedor corre con usuario no-root por defecto
- Los puertos solo exponen el necesario (8000)
- Las dependencias se instalan desde requirements.txt verificado
- Health checks automáticos para monitoreo

## 📊 Monitoreo

```bash
# Ver uso de recursos
docker stats testchatbot

# Ver logs con timestamps
docker-compose logs -f --timestamps chatbot

# Inspeccionar contenedor
docker inspect testchatbot
```

## 🚀 Producción

Para despliegue en producción, considera:

1. **Variables de entorno**: Usar archivo `.env` para configuración sensible
2. **Reverse proxy**: Usar Nginx o Traefik delante del contenedor
3. **SSL/TLS**: Configurar certificados HTTPS
4. **Logs**: Configurar log rotation y agregación
5. **Backup**: Hacer backup regular de `/data` y `/models`
6. **Resources**: Ajustar límites de CPU y memoria según necesidad

### Ejemplo con Nginx

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  chatbot:
    # ... configuración del chatbot ...
    expose:
      - "8000"
    networks:
      - internal

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - chatbot
    networks:
      - internal

networks:
  internal:
    driver: bridge
```

## 📖 Más Información

- [README.md](README.md) - Documentación completa del proyecto
- [QUICKSTART.md](QUICKSTART.md) - Guía rápida sin Docker
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Guía de uso detallada

## 🤝 Soporte

Si encuentras problemas:

1. Revisa los logs: `docker-compose logs -f`
2. Verifica el health check: `curl http://localhost:8000/health`
3. Abre un issue en GitHub con los logs relevantes
