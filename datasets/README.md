# CURZA Chatbot Dataset

Este directorio contiene el dataset masivo generado para el chatbot universitario de CURZA (UNComa).

## 📊 Contenido

### curza_dataset.json

Dataset completo con información extraída de las URLs oficiales de CURZA:

- **Preinscripción**: https://web.curza.uncoma.edu.ar/preinscripcion/
- **Oferta Académica**: https://web.curza.uncoma.edu.ar/oferta-academica
- **Trámites Académicos**: https://web.curza.uncoma.edu.ar/academica/tramites
- **Bienestar Estudiantil**: https://web.curza.uncoma.edu.ar/bienestar
- **Página Principal**: https://web.curza.uncoma.edu.ar/

## 📈 Estadísticas del Dataset

- **Total de intents**: 36
- **Total de preguntas**: 1,046
- **Promedio de preguntas por intent**: 29.1
- **Mínimo de preguntas por intent**: 29
- **Máximo de preguntas por intent**: 30

✅ **Todos los intents cumplen con el requisito mínimo de 20 variaciones**

## 🎯 Tipos de Variaciones de Preguntas

Cada intent incluye múltiples variaciones de preguntas:

### 1. Variaciones Formales
- ¿Dónde puedo consultar sobre...?
- ¿Cuál es el procedimiento para...?
- ¿En qué sección encuentro...?
- ¿Podría indicarme dónde ver...?

### 2. Variaciones Informales/Coloquiales
- ¿Dónde veo...?
- ¿Cómo hago para...?
- ¿Tenés el link de...?
- ¿Me pasás info sobre...?
- necesito...
- quiero ver...
- busco...

### 3. Errores de Tipeo Comunes (sin tildes)
- ¿Donde esta...? (en lugar de "dónde está")
- ¿Como hago...? (en lugar de "cómo hago")
- ¿Cual es...? (en lugar de "cuál es")
- ¿Que es...? (en lugar de "qué es")

### 4. Variaciones con Sinónimos
- consultar / ver / buscar / encontrar / acceder / revisar / mirar
- información / info / datos / detalles
- página / sitio / web / link / enlace / URL

### 5. Preguntas Parciales/Incompletas
- Palabras sueltas: "preinscripción", "becas", "carreras"
- Frases cortas: "info preinscripción", "quiero saber de..."
- Necesidades directas: "necesito certificado"

### 6. Variaciones Regionales Argentinas
- ¿Dónde me fijo...?
- ¿Dónde me anoto...?
- ¿Cómo tramito...?

## 📝 Estructura del Dataset

Cada entrada del dataset sigue esta estructura:

```json
{
  "intent": "nombre_unico_del_intent",
  "questions": [
    "¿Primera variación de pregunta?",
    "Segunda variación",
    "Tercera variación",
    "... (mínimo 20 variaciones)"
  ],
  "answer": "Respuesta que incluye URL específica y descripción de qué encontrará el usuario"
}
```

## 🎓 Categorías de Intents

### Preinscripción (3 intents)
- Preinscripción principal
- Requisitos de preinscripción
- Fechas de preinscripción

### Oferta Académica (5 intents)
- Oferta académica principal
- Carreras de grado
- Tecnicaturas
- Profesorados
- Planes de estudio

### Trámites Académicos (9 intents)
- Trámites principales
- Certificados generales
- Certificado de alumno regular
- Certificado analítico
- Cambio de carrera
- Equivalencias
- Pase de universidad
- Título intermedio
- Baja de materias

### Bienestar Estudiantil (7 intents)
- Bienestar principal
- Becas
- Comedor universitario
- Residencia estudiantil
- Salud estudiantil
- Apoyo psicológico
- Deportes y actividades

### Información General (8 intents)
- Página principal
- Contacto
- Ubicación
- Calendario académico
- Horarios de atención
- Autoridades
- Biblioteca
- Inscripción general

### Servicios Online (4 intents)
- Aula virtual
- Email institucional
- Inscripción a materias
- Mesa de exámenes

## 🔧 Formato de Respuestas

Todas las respuestas siguen estos principios:

1. ✅ **Incluyen el enlace específico** donde encontrar la información
2. ✅ **NO proporcionan datos específicos** (fechas, horarios, precios) que puedan quedar desactualizados
3. ✅ **Describen qué información** encontrará el usuario en ese enlace
4. ✅ **Usan lenguaje claro y cercano** (tono argentino con "vos")
5. ✅ **Son concisas pero completas**

### Ejemplo de respuesta:

```
Podés consultar las fechas de preinscripción en: https://web.curza.uncoma.edu.ar/preinscripcion/

En esta página encontrarás el cronograma actualizado con las fechas de inicio 
y cierre de la preinscripción para cada período académico.
```

## 🚀 Uso del Dataset

### Para entrenamiento del chatbot:

```python
import json

# Cargar el dataset
with open('datasets/curza_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Procesar intents
for intent in dataset:
    intent_name = intent['intent']
    questions = intent['questions']
    answer = intent['answer']
    
    # Tu código de entrenamiento aquí
    print(f"Intent: {intent_name}")
    print(f"Preguntas: {len(questions)}")
```

### Para actualizar el dataset:

```bash
# Regenerar el dataset completo
python generate_dataset.py
```

## 📋 Notas Importantes

1. **URLs actualizadas**: Todas las URLs apuntan a los sitios oficiales de CURZA
2. **Contenido dinámico**: Las respuestas redirigen al sitio web para información actualizada
3. **Escalabilidad**: El script `generate_dataset.py` puede ser modificado para agregar más intents
4. **Mantenimiento**: Revisar periódicamente que las URLs sigan siendo válidas

## 🔄 Generación Automática

El dataset fue generado automáticamente usando el script `generate_dataset.py` que:

1. Identifica temas y secciones relevantes de CURZA
2. Genera múltiples variaciones de preguntas para cada tema
3. Crea respuestas apropiadas con URLs
4. Valida que cada intent tenga al menos 20 preguntas
5. Exporta todo en formato JSON estructurado

## 📞 Contacto

Para preguntas o mejoras sobre el dataset, contactar al equipo de desarrollo.

---

**Última actualización**: 2025-12-15
**Versión**: 1.0
**Generado por**: generate_dataset.py
