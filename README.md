# TestChatBot - Chatbot Estadístico Probabilístico desde Cero

> "Dejo que Copilot haga mi trabajo creando un chatbot como prueba : 3"

## 🎯 Descripción del Proyecto

Chatbot estadístico probabilístico construido **completamente desde cero** para responder preguntas sobre información universitaria, específicamente sobre cómo navegar y encontrar información en el sitio web de la universidad.

### Características principales:
- ✅ **Modelo desde cero**: Sin usar LLMs pre-entrenados
- ✅ **Estadístico y probabilístico**: Basado en cálculos de probabilidad
- ✅ **Scope limitado**: Solo responde preguntas relacionadas con la universidad
- ✅ **Entrenamiento incremental**: Diseñado para entrenar en sesiones nocturnas
- ✅ **Escalable**: Preparado para continuar entrenamiento en servidor

---

## 🧠 Arquitectura del Modelo

### Tipo de Modelo: LSTM Bidireccional Encoder-Decoder con Attention

```
Pregunta del usuario
      ↓
[Tokenización]
      ↓
[Embedding Layer] (entrenado desde cero)
      ↓
[LSTM Encoder Bidireccional]
      ↓
[Context Vector + Attention Mechanism]
      ↓
[LSTM Decoder]
      ↓
[Capa de Salida con Softmax]
      ↓
[Sampling Probabilístico]
      ↓
Respuesta generada
```

### Componentes construidos desde cero:
1. **Tokenizador custom** - Vocabulario específico del dominio
2. **Embeddings** - Vectores de palabras entrenables
3. **LSTM Encoder** - Procesa la pregunta
4. **Attention Mechanism** - Focaliza en partes relevantes
5. **LSTM Decoder** - Genera la respuesta palabra por palabra
6. **Filtro de Scope** - Rechaza preguntas fuera del dominio

---

## 📁 Estructura del Proyecto

```
TestChatBot/
├── data/
│   ├── raw/
│   │   └── qa_dataset.txt          # Dataset con preguntas y respuestas
│   ├── processed/
│   │   ├── train.pkl               # Datos de entrenamiento procesados
│   │   ├── val.pkl                 # Datos de validación
│   │   └── vocab.json              # Vocabulario generado
│   └── scope_keywords.json         # Palabras clave del dominio
│
├── models/
│   ├── checkpoints/                # Guardado automático durante entrenamiento
│   │   ├── checkpoint_epoch_1.pt
│   │   ├── checkpoint_epoch_5.pt
│   │   └── ...
│   ├── tokenizer/
│   │   └── tokenizer.pkl           # Tokenizador entrenado
│   └── final/
│       └── best_model.pt           # Mejor modelo
│
├── src/
│   ├── preprocessing.py            # Limpieza y preparación de datos
│   ├── tokenizer.py                # Tokenizador custom
│   ├── embeddings.py               # Capa de embeddings
│   ├── encoder.py                  # LSTM Encoder
│   ├── decoder.py                  # LSTM Decoder con Attention
│   ├── model.py                    # Modelo completo Seq2Seq
│   ├── train.py                    # Script de entrenamiento
│   ├── inference.py                # Generación de respuestas
│   └── scope_filter.py             # Filtro de relevancia
│
├── app/
│   ├── api.py                      # API REST con FastAPI
│   ├── chatbot.py                  # Lógica del chatbot
│   └── static/                     # Frontend web simple
│       ├── index.html
│       ├── style.css
│       └── script.js
│
├── config.yaml                     # Configuración de hiperparámetros
├── requirements.txt                # Dependencias del proyecto
├── train.sh                        # Script para entrenar fácilmente
├── README.md                       # Este archivo
└── .gitignore
```

---

## 🛠️ Stack Tecnológico

### Lenguaje y Framework
- **Python 3.8+**
- **PyTorch** (para construir el modelo desde cero)

### Bibliotecas principales
- `torch` - Framework de deep learning
- `numpy` - Operaciones numéricas
- `nltk` - Procesamiento de lenguaje natural
- `scikit-learn` - Métricas y preprocesamiento
- `pyyaml` - Configuración
- `tqdm` - Barras de progreso
- `matplotlib` / `seaborn` - Visualización

### API y Frontend
- `fastapi` - API REST
- `uvicorn` - Servidor ASGI
- HTML/CSS/JavaScript vanilla

---

## ⚙️ Configuración del Modelo

### Hiperparámetros (config.yaml)

```yaml
model:
  embedding_dim: 256          # Dimensión de embeddings
  hidden_dim: 512             # Dimensión de LSTM
  num_layers: 2               # Capas de LSTM
  dropout: 0.3                # Dropout para regularización
  bidirectional: true         # LSTM bidireccional en encoder
  attention: true             # Usar mecanismo de attention

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100             # Entrenamientos nocturnos
  gradient_clip: 5.0
  checkpoint_every: 5         # Guardar cada 5 épocas
  early_stopping_patience: 10

data:
  train_split: 0.8
  val_split: 0.2
  max_seq_length: 50          # Longitud máxima de secuencias
  min_word_freq: 2            # Frecuencia mínima para vocabulario

scope_filter:
  similarity_threshold: 0.6   # Umbral para aceptar preguntas
  keywords_file: "data/scope_keywords.json"
```

---

## 🚀 Uso

### 1. Instalación

```bash
# Clonar repositorio
git clone https://github.com/Nnico0w0/TestChatBot.git
cd TestChatBot

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Preprocesamiento

```bash
python src/preprocessing.py
```

Esto generará:
- `data/processed/train.pkl` - Datos de entrenamiento
- `data/processed/val.pkl` - Datos de validación

### 3. Construir Tokenizador

```bash
python src/tokenizer.py
```

Esto generará:
- `models/tokenizer/tokenizer.pkl` - Tokenizador entrenado
- `data/processed/vocab.json` - Vocabulario del modelo

### 4. Entrenamiento

```bash
# Entrenamiento simple
python src/train.py

# O usar el script preparado
bash train.sh
```

**Para entrenamientos nocturnos:**
```bash
# En Linux/Mac
nohup python src/train.py > training.log 2>&1 &

# En Windows (PowerShell)
Start-Process python -ArgumentList "src/train.py" -RedirectStandardOutput "training.log" -NoNewWindow
```

### 5. Inferencia (probar el chatbot)

```bash
# Modo interactivo
python src/inference.py

# Pregunta única
python src/inference.py --question "¿Cuándo comienzan las inscripciones?"
```

### 6. Ejecutar API Web

```bash
uvicorn app.api:app --reload
# Abrir http://localhost:8000 en el navegador
```

---

## 📊 Dataset

### Formato del archivo `qa_dataset.txt`

El dataset está estructurado en formato JSON con la siguiente estructura:

```json
[
  {
    "intent": "fechas_inscripcion",
    "questions": [
      "¿Cuándo comienzan las inscripciones a la universidad?",
      "¿Cuándo abren las inscripciones?",
      "¿Cuándo puedo anotarme en la universidad?"
    ],
    "answer": "Las inscripciones comienzan en las fechas publicadas cada año por la institución..."
  }
]
```

### Intents incluidos:
- Inscripción y admisión
- Fechas importantes
- Requisitos y documentación
- Navegación del sitio web
- Información de carreras
- Consultas administrativas

---

## 🎯 Filtro de Scope

El chatbot incluye un **filtro de relevancia** que:

1. ✅ Calcula la similitud semántica entre la pregunta y el dominio
2. ✅ Compara con palabras clave del scope universitario
3. ✅ Rechaza cortésmente preguntas fuera del tema

**Ejemplo:**
```
Usuario: "¿Cuál es la capital de Francia?"
Bot: "Lo siento, solo puedo responder preguntas sobre información 
     universitaria y navegación del sitio web de la institución."
```

---

## 📈 Sistema de Checkpoints

Durante el entrenamiento nocturno, el modelo guarda automáticamente:

- ✅ **Checkpoint cada N épocas** (configurable)
- ✅ **Mejor modelo** según pérdida de validación
- ✅ **Métricas de entrenamiento** (loss, accuracy, perplexity)
- ✅ **Estado del optimizador** (para continuar entrenamiento)

Si el entrenamiento se interrumpe, puedes continuar desde el último checkpoint:

```bash
python src/train.py --resume --checkpoint models/checkpoints/checkpoint_epoch_20.pt
```

---

## 🧪 Evaluación del Modelo

### Métricas utilizadas:
- **Perplexity** - Mide qué tan "sorprendido" está el modelo
- **Loss de validación** - Error en datos no vistos

---

## 🔬 Fundamentos Teóricos

### ¿Por qué es "estadístico probabilístico"?

1. **Embeddings probabilísticos**: Cada palabra se representa como un vector en un espacio de probabilidades

2. **LSTM calcula probabilidades**: En cada paso temporal, calcula la probabilidad de la siguiente palabra dado el contexto

3. **Softmax**: Convierte las salidas en distribución de probabilidad sobre todo el vocabulario

4. **Sampling**: La respuesta se genera muestreando de la distribución de probabilidad (no es determinístico)

### Ecuaciones clave:

**LSTM Cell:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # Input gate
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # Output gate
c_t = f_t * c_{t-1} + i_t * tanh(W_c · [h_{t-1}, x_t] + b_c)
h_t = o_t * tanh(c_t)
```

**Attention:**
```
score(h_t, h_s) = h_t^T · W_a · h_s
α_t = softmax(score(h_t, h_s))
context_t = Σ(α_t * h_s)
```

---

## 🖥️ Requisitos de Hardware

### Mínimo (CPU):
- 8GB RAM
- Entrenamiento lento (días)

### Recomendado (GPU):
- ASUS TUF A15 o similar
- GPU NVIDIA (RTX 3050+)
- 16GB RAM
- Entrenamiento: 2-7 noches

### Óptimo (Servidor):
- GPU NVIDIA (RTX 3080+)
- 32GB+ RAM
- Entrenamiento: horas

---

## 🤝 Contribuciones

Este es un proyecto educativo. Sugerencias y mejoras son bienvenidas.

---

## 📄 Licencia

MIT License

---

## 👤 Autor

**Nnico0w0**

GitHub: [@Nnico0w0](https://github.com/Nnico0w0)

---

## 📚 Referencias

### Papers y recursos:
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## 🎓 Notas del Proyecto

Este chatbot es un proyecto educativo que demuestra:
- Construcción de modelos de NLP desde cero
- Arquitecturas Encoder-Decoder
- Mecanismos de Attention
- Entrenamiento de modelos secuenciales
- Filtrado de dominio
- Despliegue de modelos

**Objetivo**: Aprender los fundamentos de los chatbots estadísticos probabilísticos sin depender de modelos pre-entrenados, construyendo cada componente desde cero para entender profundamente cómo funcionan.

---

**¡Empecemos a entrenar!** 🚀
