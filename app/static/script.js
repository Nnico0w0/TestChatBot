// Script para manejar la interacción del chatbot

const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

// Event listeners
sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Función para enviar mensaje
async function sendMessage() {
    const question = userInput.value.trim();
    
    if (!question) {
        return;
    }
    
    // Añadir mensaje del usuario
    addMessage(question, 'user');
    
    // Limpiar input
    userInput.value = '';
    
    // Deshabilitar input mientras se procesa
    setInputEnabled(false);
    
    // Mostrar indicador de escritura
    showTypingIndicator();
    
    try {
        // Llamar a la API
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                max_length: 50
            })
        });
        
        if (!response.ok) {
            throw new Error('Error en la respuesta del servidor');
        }
        
        const data = await response.json();
        
        // Remover indicador de escritura
        removeTypingIndicator();
        
        // Añadir respuesta del bot
        addMessage(data.answer, 'bot');
        
    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator();
        addMessage('Lo siento, hubo un error al procesar tu pregunta. Por favor, intenta nuevamente.', 'bot');
    } finally {
        setInputEnabled(true);
        userInput.focus();
    }
}

// Función para añadir mensaje al chat
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll al final
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Función para mostrar indicador de escritura
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message';
    typingDiv.id = 'typing-indicator';
    
    const indicatorDiv = document.createElement('div');
    indicatorDiv.className = 'message-content typing-indicator';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        indicatorDiv.appendChild(dot);
    }
    
    typingDiv.appendChild(indicatorDiv);
    chatMessages.appendChild(typingDiv);
    
    // Scroll al final
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Función para remover indicador de escritura
function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Función para habilitar/deshabilitar input
function setInputEnabled(enabled) {
    userInput.disabled = !enabled;
    sendButton.disabled = !enabled;
}

// Verificar estado del servidor al cargar
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (!data.model_loaded) {
            console.warn('El modelo aún no está cargado');
        }
    } catch (error) {
        console.error('Error al verificar el estado del servidor:', error);
    }
});
