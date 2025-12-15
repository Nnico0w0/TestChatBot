# Guía de Generación de Dataset CURZA

Esta guía explica cómo usar el sistema de generación de dataset masivo para el chatbot universitario de CURZA.

## 📋 Tabla de Contenidos

1. [Requisitos](#requisitos)
2. [Instalación](#instalación)
3. [Uso Básico](#uso-básico)
4. [Estructura del Dataset](#estructura-del-dataset)
5. [Personalización](#personalización)
6. [Validación](#validación)
7. [Solución de Problemas](#solución-de-problemas)

## 🔧 Requisitos

### Dependencias
- Python 3.8+
- beautifulsoup4>=4.12.3
- lxml>=5.1.0
- requests>=2.31.0

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

O instalar solo las dependencias necesarias para generación de dataset:

```bash
pip install beautifulsoup4==4.12.3 lxml==5.1.0 requests>=2.31.0
```

## 🚀 Uso Básico

### Generar el Dataset Completo

```bash
python generate_dataset.py
```

Este comando:
1. Procesa todas las URLs de CURZA configuradas
2. Genera múltiples variaciones de preguntas para cada tema
3. Crea respuestas con URLs apropiadas
4. Guarda el dataset en `datasets/curza_dataset.json`

### Validar el Dataset Generado

```bash
python validate_dataset.py
```

O validar un archivo específico:

```bash
python validate_dataset.py path/to/dataset.json
```

## 📊 Estructura del Dataset

### Formato JSON

Cada entrada en el dataset sigue esta estructura:

```json
{
  "intent": "nombre_del_intent",
  "questions": [
    "¿Pregunta 1?",
    "¿Pregunta 2?",
    "pregunta 3",
    ...
  ],
  "answer": "Respuesta con URL: https://... y descripción de contenido"
}
```

### Ejemplo Completo

```json
{
  "intent": "preinscripcion_principal",
  "questions": [
    "¿Cómo me preinscribo?",
    "¿Dónde está la preinscripción?",
    "Necesito preinscribirme",
    "¿Donde esta la preinscripcion?",
    "¿Me pasás info sobre preinscripción?",
    ...
  ],
  "answer": "Podés realizar la preinscripción en el siguiente enlace: https://web.curza.uncoma.edu.ar/preinscripcion/\n\nEn esta página encontrarás toda la información sobre el proceso de preinscripción..."
}
```

## 🎨 Personalización

### Agregar Nuevas URLs

Edita `generate_dataset.py` y modifica la lista de URLs:

```python
self.urls = [
    "https://web.curza.uncoma.edu.ar/preinscripcion/",
    "https://web.curza.uncoma.edu.ar/oferta-academica",
    # Agregar más URLs aquí
    "https://web.curza.uncoma.edu.ar/nueva-seccion",
]
```

### Crear Nuevos Generadores de Intents

Agrega un nuevo método en la clase `CURZADatasetGenerator`:

```python
def generate_mi_seccion_intents(self, url: str) -> List[Dict]:
    """Generate intents for mi sección."""
    intents = []
    
    # Intent principal
    questions = self.generate_questions_variations(
        "mi tema",
        [
            "¿Pregunta base 1?",
            "¿Pregunta base 2?",
        ]
    )
    
    intents.append({
        "intent": "mi_tema_principal",
        "questions": questions,
        "answer": (
            f"Podés consultar sobre mi tema en: {url}\n\n"
            "Descripción de qué encontrará el usuario."
        )
    })
    
    return intents
```

Luego registra el generador en `generate_dataset()`:

```python
url_generators = {
    # ... existentes ...
    "https://web.curza.uncoma.edu.ar/mi-seccion": self.generate_mi_seccion_intents,
}
```

### Modificar Variaciones de Preguntas

Las variaciones se generan automáticamente en `generate_questions_variations()`. 

Para agregar más tipos de variaciones, edita los templates en ese método:

```python
# Agregar nuevos templates
nuevas_variaciones = [
    f"template 1 {topic}",
    f"template 2 {topic}",
]

all_templates.extend(nuevas_variaciones)
```

## ✅ Validación

### Script de Validación

El script `validate_dataset.py` verifica:

- ✅ Estructura JSON válida
- ✅ Campos requeridos presentes (intent, questions, answer)
- ✅ Mínimo 20 preguntas por intent
- ✅ No hay preguntas vacías o duplicadas
- ✅ Todas las respuestas contienen URLs
- ✅ Tipos de datos correctos

### Ejecutar Validación

```bash
python validate_dataset.py
```

### Salida de Validación

```
================================================================================
VALIDATING CURZA DATASET
================================================================================

✓ Successfully loaded dataset from datasets/curza_dataset.json
✓ Dataset contains 36 intents

================================================================================
STATISTICS
================================================================================
Total intents: 36
Total questions: 1046
Average questions per intent: 29.1
...

✅ VALIDATION PASSED
Dataset is ready for use!
```

## 🔧 Solución de Problemas

### Error: "Module not found: beautifulsoup4"

**Solución:**
```bash
pip install beautifulsoup4 lxml
```

### Error: "Failed to fetch URL: Status 403"

**Causa:** El servidor puede estar bloqueando requests automáticos.

**Solución:**
- Verifica que la URL sea accesible en un navegador
- El script ya incluye un User-Agent, pero algunos sitios pueden requerir más medidas
- Considera agregar delays más largos entre requests

### Dataset generado con menos de 20 preguntas por intent

**Causa:** El generador de variaciones puede no estar funcionando correctamente.

**Solución:**
- Verifica que `generate_questions_variations()` esté generando suficientes templates
- Agrega más base_questions en los métodos de generación de intents
- Ejecuta con debug para ver cuántas variaciones se generan

### Advertencia: "Intent answer does not contain a URL"

**Causa:** La respuesta no incluye un enlace.

**Solución:**
Asegúrate de que todas las respuestas incluyan una URL en el formato:

```python
answer = (
    f"Podés consultar ... en: {url}\n\n"
    "Descripción..."
)
```

## 📖 Uso del Dataset Generado

### Cargar el Dataset

```python
import json

with open('datasets/curza_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Iterar sobre intents
for intent in dataset:
    print(f"Intent: {intent['intent']}")
    print(f"Preguntas: {len(intent['questions'])}")
    print(f"Respuesta: {intent['answer'][:100]}...")
    print()
```

### Integrar con el Chatbot

El dataset puede ser usado directamente para entrenamiento del modelo:

```python
# Preparar datos de entrenamiento
training_data = []

for intent in dataset:
    intent_name = intent['intent']
    answer = intent['answer']
    
    for question in intent['questions']:
        training_data.append({
            'question': question,
            'answer': answer,
            'intent': intent_name
        })

print(f"Total training samples: {len(training_data)}")
```

### Exportar a Otros Formatos

#### CSV
```python
import csv

with open('dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Intent', 'Question', 'Answer'])
    
    for intent in dataset:
        for question in intent['questions']:
            writer.writerow([intent['intent'], question, intent['answer']])
```

#### TXT (formato del proyecto)
```python
with open('dataset.txt', 'w', encoding='utf-8') as f:
    for intent in dataset:
        f.write(f"\nIntent: {intent['intent']}\n")
        f.write("Questions:\n")
        for q in intent['questions']:
            f.write(f"  - {q}\n")
        f.write(f"\nAnswer:\n{intent['answer']}\n")
        f.write("\n" + "="*80 + "\n")
```

## 📈 Métricas de Calidad

### Verificar Cobertura

```python
# Contar tipos de variaciones
formal_count = 0
informal_count = 0
typo_count = 0

for intent in dataset:
    for q in intent['questions']:
        if '¿Dónde puedo' in q or '¿Cuál es' in q:
            formal_count += 1
        elif 'necesito' in q.lower() or 'quiero' in q.lower():
            informal_count += 1
        elif 'donde esta' in q.lower() or 'como hago' in q.lower():
            typo_count += 1

print(f"Formal: {formal_count}")
print(f"Informal: {informal_count}")
print(f"Con typos: {typo_count}")
```

### Analizar Distribución

```python
import matplotlib.pyplot as plt

# Distribución de preguntas por intent
question_counts = [len(intent['questions']) for intent in dataset]

plt.figure(figsize=(10, 6))
plt.hist(question_counts, bins=20, edgecolor='black')
plt.xlabel('Número de preguntas')
plt.ylabel('Frecuencia')
plt.title('Distribución de preguntas por intent')
plt.savefig('distribution.png')
```

## 🔄 Actualización del Dataset

### Regenerar Completamente

```bash
# Backup del dataset actual
cp datasets/curza_dataset.json datasets/curza_dataset.backup.json

# Generar nuevo dataset
python generate_dataset.py

# Validar
python validate_dataset.py
```

### Fusionar con Dataset Existente

```python
import json

# Cargar datasets
with open('datasets/curza_dataset.json', 'r') as f:
    dataset_curza = json.load(f)

with open('data/raw/qa_dataset.txt', 'r') as f:
    # Cargar dataset original (si está en JSON)
    pass  # Implementar según formato

# Fusionar evitando duplicados
intents_dict = {intent['intent']: intent for intent in dataset_curza}

# Guardar fusionado
with open('datasets/merged_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(list(intents_dict.values()), f, ensure_ascii=False, indent=2)
```

## 📚 Referencias

- [Documentación del Dataset](datasets/README.md)
- [Script de Generación](generate_dataset.py)
- [Script de Validación](validate_dataset.py)
- [URLs de CURZA](https://web.curza.uncoma.edu.ar/)

## 🆘 Soporte

Para problemas, mejoras o preguntas:
1. Revisa esta guía y el README del dataset
2. Ejecuta el script de validación para identificar problemas
3. Verifica los logs de ejecución del script de generación
4. Contacta al equipo de desarrollo

---

**Última actualización**: 2025-12-15
**Versión**: 1.0
