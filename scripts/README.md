# Scripts - Utilidades para TestChatBot

Colección de scripts útiles para gestión y optimización de datasets.

## 📜 Scripts Disponibles

### `unify_datasets.py`

Script para unificar y optimizar múltiples datasets del chatbot.

**Funcionalidad:**
- ✅ Lee múltiples datasets (JSON y formato texto mixto)
- ✅ Elimina duplicados de intents
- ✅ Combina preguntas de intents duplicados
- ✅ Optimiza respuestas (30-150 palabras)
- ✅ Limpia placeholders ([ENLACE], [EMAIL], etc.)
- ✅ Valida formato JSON
- ✅ Genera reporte de estadísticas
- ✅ Identifica intents con pocas preguntas (<15)

**Uso:**

```bash
# Desde la raíz del proyecto
python scripts/unify_datasets.py
```

**Archivos de entrada:**
- `data/raw/qa_dataset.txt` - Dataset en formato texto mixto (existente)
- `datasets/curza_dataset.json` - Dataset JSON de CURZA (opcional)

**Archivo de salida:**
- `data/raw/unified_dataset.json` - Dataset unificado y optimizado

**Estructura del JSON de salida:**

```json
[
  {
    "intent": "nombre_unico_del_intent",
    "questions": [
      "¿Pregunta 1?",
      "¿Pregunta 2?",
      "¿Pregunta 3?"
    ],
    "answer": "Respuesta concisa y optimizada..."
  }
]
```

**Ejemplo de salida:**

```
🚀 Iniciando unificación de datasets

📂 Parseando archivo de texto: data/raw/qa_dataset.txt
  ✅ 16 intents extraídos
📂 Cargando JSON desde: datasets/curza_dataset.json
⚠️  Archivo no encontrado: datasets/curza_dataset.json

🔄 Unificando datasets...
  ✅ 16 intents únicos
  🗑️  0 duplicados eliminados

⚙️  Optimizando dataset...
  ✅ Dataset optimizado

✓ Validando dataset...
  ✅ Dataset válido

💾 Guardando dataset en: data/raw/unified_dataset.json
  ✅ Dataset guardado exitosamente
  📁 Tamaño: 7.2 KB

============================================================
📊 REPORTE DE UNIFICACIÓN
============================================================
Total de intents: 16
Total de preguntas: 60
Promedio de preguntas por intent: 3.8
Duplicados eliminados: 0
============================================================

✅ Proceso completado exitosamente!
```

## 🔧 Integración con src/preprocessing.py

El script `src/preprocessing.py` ha sido actualizado para soportar lectura directa de archivos JSON, manteniendo compatibilidad con el formato texto mixto anterior.

**Para usar el dataset unificado:**

1. Ejecutar el script de unificación:
   ```bash
   python scripts/unify_datasets.py
   ```

2. Actualizar `config.yaml`:
   ```yaml
   data:
     raw_data_path: "data/raw/unified_dataset.json"
   ```

3. Ejecutar preprocesamiento normalmente:
   ```bash
   python src/preprocessing.py
   ```

## 📝 Agregar Nuevos Datasets

Para agregar un nuevo dataset (por ejemplo, CURZA):

1. Crear el archivo en formato JSON:
   ```bash
   mkdir -p datasets
   # Copiar o crear datasets/curza_dataset.json
   ```

2. El formato debe ser:
   ```json
   [
     {
       "intent": "nombre_intent",
       "questions": ["pregunta1", "pregunta2"],
       "answer": "respuesta"
     }
   ]
   ```

3. Ejecutar el script de unificación:
   ```bash
   python scripts/unify_datasets.py
   ```

El script automáticamente detectará y procesará el nuevo dataset.

## 🎯 Criterios de Calidad

El script aplica los siguientes criterios:

- ✅ Respuestas entre 30-150 palabras
- ✅ Elimina placeholders genéricos ([ENLACE], [EMAIL], etc.)
- ✅ Mantiene solo información esencial
- ✅ Elimina duplicados de preguntas dentro de cada intent
- ✅ Combina intents duplicados preservando la respuesta más completa
- ✅ Valida estructura JSON

## 🐛 Solución de Problemas

### Dataset no encontrado

Si el script no encuentra un dataset:
```
⚠️  Archivo no encontrado: datasets/curza_dataset.json
```

El script continúa procesando los datasets disponibles.

### Intents con pocas preguntas

El reporte identifica intents que tienen menos de 15 preguntas:
```
⚠️  Intents con menos de 15 preguntas:
  - intent_ejemplo: 8 preguntas
```

Considera agregar más variaciones de preguntas para mejorar el entrenamiento.

### Errores de formato JSON

Si un bloque JSON no se puede parsear, el script:
- Registra el error
- Continúa con el siguiente bloque
- No detiene el proceso

## 📚 Más Información

Para más detalles sobre el formato de datasets y preprocesamiento, consultar:
- [README.md](../README.md) - Documentación principal
- [src/preprocessing.py](../src/preprocessing.py) - Código de preprocesamiento
- [config.yaml](../config.yaml) - Configuración del proyecto
