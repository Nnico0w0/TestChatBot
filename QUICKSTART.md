# 🚀 QuickStart Guide - TestChatBot

Guía rápida para empezar en 5 minutos.

## ⚡ Opción 1: Con Docker (Más Rápido) 🐳

```bash
# 1. Clonar y entrar al directorio
git clone https://github.com/Nnico0w0/TestChatBot.git
cd TestChatBot

# 2. Ejecutar con el script de inicio rápido
bash docker-start.sh

# La aplicación estará disponible en http://localhost:8000
```

Ver [DOCKER.md](DOCKER.md) para más detalles y opciones avanzadas.

## ⚡ Opción 2: Instalación Local

```bash
# 1. Clonar y entrar al directorio
git clone https://github.com/Nnico0w0/TestChatBot.git
cd TestChatBot

# 2. Crear entorno virtual e instalar
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Preparar datos
python src/preprocessing.py
python src/tokenizer.py

# 4. Entrenar (ejemplo rápido con 10 épocas)
PYTHONPATH=. python src/train.py --epochs 10

# 5. Probar el chatbot
PYTHONPATH=. python src/inference.py
```

## 💬 Comandos Principales

### Preprocesamiento
```bash
python src/preprocessing.py
```

### Tokenizador
```bash
python src/tokenizer.py
```

### Entrenamiento
```bash
# Entrenamiento básico
PYTHONPATH=. python src/train.py

# Personalizado
PYTHONPATH=. python src/train.py --epochs 50

# Background (Linux/Mac)
nohup python src/train.py > training.log 2>&1 &
```

### Inferencia
```bash
# Modo interactivo
PYTHONPATH=. python src/inference.py

# Pregunta única
PYTHONPATH=. python src/inference.py --question "tu pregunta"
```

### API Web
```bash
# Iniciar servidor
uvicorn app.api:app --reload

# Acceder
# Browser: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## 📁 Estructura Simplificada

```
TestChatBot/
├── data/
│   ├── raw/qa_dataset.txt       # Dataset original
│   ├── processed/               # Datos procesados (auto-generados)
│   └── scope_keywords.json      # Keywords del dominio
├── models/
│   ├── checkpoints/             # Checkpoints (auto-generados)
│   ├── tokenizer/               # Tokenizador (auto-generado)
│   └── final/best_model.pt      # Mejor modelo (auto-generado)
├── src/
│   ├── preprocessing.py         # Paso 1: Preprocesar
│   ├── tokenizer.py            # Paso 2: Tokenizar
│   ├── train.py                # Paso 3: Entrenar
│   └── inference.py            # Paso 4: Inferir
├── app/
│   ├── api.py                  # API REST
│   └── static/                 # Frontend web
├── config.yaml                 # Configuración
└── requirements.txt            # Dependencias
```

## 🎯 Workflow Típico

```
1. Preprocesar datos
   ↓
2. Construir tokenizador
   ↓
3. Entrenar modelo (varias épocas)
   ↓
4. Probar con inferencia
   ↓
5. Desplegar API (opcional)
```

## ⚙️ Configuración Básica

Editar `config.yaml`:

```yaml
training:
  num_epochs: 100        # Épocas de entrenamiento
  batch_size: 32         # Tamaño de batch
  learning_rate: 0.001   # Tasa de aprendizaje

model:
  hidden_dim: 512        # Dimensión oculta
  embedding_dim: 256     # Dimensión embeddings
```

## 🐛 Soluciones Rápidas

**Error de módulo:**
```bash
PYTHONPATH=. python src/tu_script.py
```

**Modelo no encontrado:**
```bash
# Entrenar primero
PYTHONPATH=. python src/train.py --epochs 10
```

**NLTK recursos:**
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

## 📊 Ejemplo de Uso Completo

```bash
# Terminal 1: Entrenar
PYTHONPATH=. python src/train.py --epochs 20

# Terminal 2: Monitorear (mientras entrena)
watch -n 5 'ls -lh models/checkpoints/'

# Después del entrenamiento: Probar
PYTHONPATH=. python src/inference.py --question "¿Cuándo comienzan las inscripciones?"

# Iniciar API web
uvicorn app.api:app --reload
```

## 📝 Preguntas de Ejemplo

**Dentro del scope (aceptadas):**
- "¿Cuándo comienzan las inscripciones?"
- "¿Dónde encuentro el calendario académico?"
- "¿Qué carreras ofrece la universidad?"
- "¿Cómo me contacto con el Departamento de Alumnos?"

**Fuera del scope (rechazadas):**
- "¿Cuál es la capital de Francia?"
- "¿Quién ganó el mundial de fútbol?"
- "¿Cómo se hace una pizza?"

## 🎓 Recursos

- **README completo**: `README.md`
- **Guía detallada**: `USAGE_GUIDE.md`
- **Dataset**: `data/raw/qa_dataset.txt`
- **Configuración**: `config.yaml`

---

**¡Listo para empezar!** 🚀

Para más detalles, consulta `USAGE_GUIDE.md` o `README.md`.
