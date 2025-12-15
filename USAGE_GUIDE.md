# Guía de Uso - TestChatBot

Esta guía explica cómo usar el chatbot desde la instalación hasta el despliegue.

## 📋 Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- 8GB RAM mínimo (16GB recomendado)
- GPU NVIDIA (opcional, pero recomendado para entrenamiento)

## 🚀 Instalación Rápida

### 1. Clonar el Repositorio

```bash
git clone https://github.com/Nnico0w0/TestChatBot.git
cd TestChatBot
```

### 2. Crear Entorno Virtual

```bash
# En Linux/Mac
python -m venv venv
source venv/bin/activate

# En Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

## 📊 Preparar los Datos

### Paso 1: Preprocesamiento

```bash
python src/preprocessing.py
```

**Salida esperada:**
```
Iniciando preprocesamiento...
Leyendo dataset desde: data/raw/qa_dataset.txt
Intents encontrados: 16
Pares de Q&A creados: 60
Datos de entrenamiento: 48
Datos de validación: 12
Datos guardados en: data/processed
Preprocesamiento completado exitosamente!
```

### Paso 2: Construir Tokenizador

```bash
python src/tokenizer.py
```

**Salida esperada:**
```
Construyendo vocabulario...
Total de palabras únicas: 253
Tamaño del vocabulario: 199
Tokenizador guardado en: models/tokenizer/tokenizer.pkl
Vocabulario guardado en: data/processed/vocab.json
Tokenizador construido exitosamente!
```

## 🎯 Entrenar el Modelo

### Entrenamiento Básico

```bash
# Usando el script
bash train.sh

# O directamente con Python
PYTHONPATH=. python src/train.py
```

### Entrenamiento Personalizado

```bash
# Entrenar por N épocas específicas
PYTHONPATH=. python src/train.py --epochs 50

# Con configuración personalizada
PYTHONPATH=. python src/train.py --config config_custom.yaml
```

### Entrenamiento Nocturno (Linux/Mac)

```bash
# Ejecutar en background con logging
nohup python src/train.py > training.log 2>&1 &

# Ver el progreso
tail -f training.log
```

### Entrenamiento Nocturno (Windows)

```powershell
# En PowerShell
Start-Process python -ArgumentList "src/train.py" -RedirectStandardOutput "training.log" -NoNewWindow

# Ver el log
Get-Content training.log -Wait
```

### Salida Durante el Entrenamiento

```
Usando dispositivo: cpu
Cargando tokenizer...
Cargando datos...
Creando modelo...

Iniciando entrenamiento por 100 épocas...
Tamaño del vocabulario: 199
Datos de entrenamiento: 48
Datos de validación: 12

Época 1: 100%|██████████| 2/2 [00:05<00:00,  2.79s/it]
Época 1/100
Train Loss: 5.2627 | Train Perplexity: 192.9982
Val Loss: 4.9158 | Val Perplexity: 136.4233
Mejor modelo guardado: models/final/best_model.pt

...
```

## 💬 Usar el Chatbot

### Modo Interactivo (Consola)

```bash
PYTHONPATH=. python src/inference.py
```

**Ejemplo de uso:**
```
==================================================
Chatbot Universitario - Modo Interactivo
==================================================
Escribe 'salir' o 'exit' para terminar

Tú: ¿Cuándo comienzan las inscripciones?
Bot: Las inscripciones comienzan en las fechas publicadas cada año por la institución...

Tú: ¿Cuál es la capital de Francia?
Bot: Lo siento, solo puedo responder preguntas sobre información universitaria...

Tú: salir
¡Hasta luego!
```

### Pregunta Única

```bash
PYTHONPATH=. python src/inference.py --question "¿Dónde puedo ver el calendario académico?"
```

## 🌐 API Web

### Iniciar el Servidor

```bash
# Con uvicorn directamente
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# O usando Python
PYTHONPATH=. python app/api.py
```

### Acceder a la Interfaz Web

Abrir en el navegador: `http://localhost:8000`

### Documentación de la API

Swagger UI: `http://localhost:8000/docs`

### Usar la API

#### Endpoint POST /chat

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuándo comienzan las inscripciones?",
    "max_length": 50
  }'
```

**Respuesta:**
```json
{
  "question": "¿Cuándo comienzan las inscripciones?",
  "answer": "Las inscripciones comienzan en las fechas publicadas...",
  "in_scope": true,
  "confidence": 1.0
}
```

#### Endpoint GET /health

```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🔧 Configuración Avanzada

### Modificar Hiperparámetros

Editar `config.yaml`:

```yaml
model:
  embedding_dim: 256      # Aumentar para mejor representación
  hidden_dim: 512         # Aumentar para mayor capacidad
  num_layers: 2           # Más capas = más complejidad
  dropout: 0.3            # Reducir si hay underfitting

training:
  batch_size: 32          # Ajustar según RAM disponible
  learning_rate: 0.001    # Reducir si el loss oscila mucho
  num_epochs: 100         # Aumentar para mejor convergencia
```

### Agregar Nuevas Keywords

Editar `data/scope_keywords.json`:

```json
{
  "keywords": [
    "universidad",
    "inscripción",
    "tu_nueva_keyword"
  ],
  "rejection_messages": [
    "Tu mensaje personalizado de rechazo"
  ]
}
```

### Expandir el Dataset

Editar `data/raw/qa_dataset.txt` y agregar nuevos intents:

```json
[
  {
    "intent": "nuevo_intent",
    "questions": [
      "¿Pregunta ejemplo 1?",
      "¿Pregunta ejemplo 2?"
    ],
    "answer": "Respuesta para este intent."
  }
]
```

Luego, re-ejecutar preprocesamiento y tokenizador:

```bash
python src/preprocessing.py
python src/tokenizer.py
```

## 📈 Monitoreo del Entrenamiento

### Checkpoints

Los checkpoints se guardan automáticamente en `models/checkpoints/`:
- `checkpoint_epoch_5.pt`
- `checkpoint_epoch_10.pt`
- etc.

### Mejor Modelo

El mejor modelo (menor pérdida de validación) se guarda en:
- `models/final/best_model.pt`

### Continuar Entrenamiento (TODO)

Para continuar desde un checkpoint:

```bash
PYTHONPATH=. python src/train.py --resume --checkpoint models/checkpoints/checkpoint_epoch_20.pt
```

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'src'"

**Solución:** Usar `PYTHONPATH=.` antes del comando:

```bash
PYTHONPATH=. python src/train.py
```

### Error: "Resource punkt_tab not found"

**Solución:** El tokenizador automáticamente descarga los recursos necesarios. Si persiste:

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

### Error: "Model not found"

**Solución:** Asegurarse de entrenar el modelo primero:

```bash
PYTHONPATH=. python src/train.py --epochs 10
```

### Respuestas Incoherentes

**Causa:** El modelo necesita más entrenamiento.

**Solución:** 
1. Aumentar el número de épocas
2. Expandir el dataset
3. Ajustar hiperparámetros

### Out of Memory (OOM)

**Solución:**
1. Reducir `batch_size` en `config.yaml`
2. Reducir `hidden_dim` o `embedding_dim`
3. Usar GPU con más memoria

## 📦 Despliegue

### Docker (TODO)

```bash
# Construir imagen
docker build -t testchatbot .

# Ejecutar contenedor
docker run -p 8000:8000 testchatbot
```

### Servidor de Producción

```bash
# Instalar gunicorn
pip install gunicorn

# Ejecutar con gunicorn
gunicorn app.api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

## 📝 Notas Importantes

1. **Primera Ejecución**: El primer entrenamiento puede ser lento mientras PyTorch configura el backend.

2. **GPU vs CPU**: El entrenamiento en CPU es funcional pero lento. GPU es altamente recomendada para entrenamientos largos.

3. **Dataset Pequeño**: El dataset actual es pequeño (60 pares Q&A). Para mejores resultados, expandir a 500+ pares.

4. **Modelo Pre-entrenado**: Este proyecto NO usa modelos pre-entrenados. Todo se entrena desde cero.

5. **Scope Filter**: El filtro es simple pero efectivo. Para mayor precisión, considerar usar embeddings semánticos.

## 🎓 Aprendizaje

Este proyecto es educativo. Los conceptos principales implementados:

- ✅ Tokenización y vocabulario
- ✅ Embeddings de palabras
- ✅ Arquitectura Encoder-Decoder
- ✅ Mecanismo de Attention
- ✅ LSTM bidireccional
- ✅ Training loop con validación
- ✅ Checkpointing
- ✅ Early stopping
- ✅ API REST
- ✅ Filtrado de dominio

## 📞 Soporte

Para preguntas o problemas:
- Abrir un issue en GitHub
- Revisar la documentación en README.md
- Verificar los logs de entrenamiento

---

**¡Buena suerte con tu chatbot!** 🚀
