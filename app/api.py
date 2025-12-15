"""
API REST con FastAPI para el chatbot.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
import uvicorn

from app.chatbot import ChatBotService


# Modelos de datos
class QuestionRequest(BaseModel):
    """Modelo para la solicitud de pregunta."""
    question: str
    max_length: int = 50


class QuestionResponse(BaseModel):
    """Modelo para la respuesta."""
    question: str
    answer: str
    in_scope: bool
    confidence: float


# Crear aplicación
app = FastAPI(
    title="TestChatBot API",
    description="API REST para el chatbot estadístico probabilístico",
    version="1.0.0"
)

# Servicio del chatbot
chatbot_service = ChatBotService()


@app.on_event("startup")
async def startup_event():
    """Evento de inicio - carga el modelo."""
    try:
        print("Iniciando servidor...")
        print("Cargando modelo del chatbot...")
        chatbot_service.load_model()
        print("¡Servidor listo!")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("El modelo se cargará al recibir la primera petición.")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Sirve la página principal."""
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path)
    else:
        return """
        <html>
            <head>
                <title>TestChatBot</title>
            </head>
            <body>
                <h1>TestChatBot API</h1>
                <p>API funcionando correctamente.</p>
                <p>Documentación disponible en <a href="/docs">/docs</a></p>
            </body>
        </html>
        """


@app.get("/health")
async def health_check():
    """Endpoint de health check."""
    return {
        "status": "healthy",
        "model_loaded": chatbot_service.model is not None
    }


@app.post("/chat", response_model=QuestionResponse)
async def chat(request: QuestionRequest):
    """
    Endpoint para hacer preguntas al chatbot.
    
    Args:
        request: Pregunta del usuario
        
    Returns:
        Respuesta del chatbot
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
        
        response = chatbot_service.get_response(
            request.question,
            max_length=request.max_length
        )
        
        return QuestionResponse(**response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/ask")
async def ask_simple(question: str):
    """
    Endpoint simplificado para hacer preguntas.
    
    Args:
        question: Pregunta del usuario
        
    Returns:
        Respuesta del chatbot
    """
    try:
        response = chatbot_service.get_response(question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Montar archivos estáticos
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


def main():
    """Función principal para ejecutar el servidor."""
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
