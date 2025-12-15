"""
Script de inferencia para generar respuestas del chatbot.
"""

import argparse
import torch
import yaml
from pathlib import Path

from src.model import Seq2SeqModel
from src.tokenizer import CustomTokenizer
from src.scope_filter import ScopeFilter


class ChatBot:
    """Chatbot para inferencia."""
    
    def __init__(self, model_path: str = None, config_path: str = "config.yaml"):
        """
        Inicializa el chatbot.
        
        Args:
            model_path: Ruta al modelo entrenado
            config_path: Ruta al archivo de configuración
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Cargar tokenizer
        print("Cargando tokenizer...")
        self.tokenizer = CustomTokenizer.load()
        
        # Cargar filtro de scope
        print("Cargando filtro de scope...")
        self.scope_filter = ScopeFilter(config_path)
        
        # Cargar modelo
        if model_path is None:
            model_path = self.config['paths']['best_model_path']
        
        print(f"Cargando modelo desde: {model_path}")
        self.model = self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("¡Chatbot listo!")
    
    def _load_model(self, model_path):
        """Carga el modelo desde un checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Crear modelo
        model = Seq2SeqModel(
            vocab_size=len(self.tokenizer),
            embedding_dim=checkpoint['config']['model']['embedding_dim'],
            hidden_dim=checkpoint['config']['model']['hidden_dim'],
            num_layers=checkpoint['config']['model']['num_layers'],
            dropout=checkpoint['config']['model']['dropout'],
            bidirectional=checkpoint['config']['model']['bidirectional'],
            use_attention=checkpoint['config']['model']['attention'],
            padding_idx=0
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def predict(self, question: str, max_length: int = 50) -> str:
        """
        Genera una respuesta para la pregunta.
        
        Args:
            question: Pregunta del usuario
            max_length: Longitud máxima de la respuesta
            
        Returns:
            Respuesta generada
        """
        # Verificar scope
        is_valid, rejection_msg = self.scope_filter.filter_question(question)
        if not is_valid:
            return rejection_msg
        
        # Preprocesar pregunta
        question = question.lower().strip()
        
        # Codificar
        question_ids = self.tokenizer.encode(question, add_special_tokens=True)
        question_tensor = torch.tensor([question_ids], dtype=torch.long).to(self.device)
        
        # Generar respuesta
        with torch.no_grad():
            generated_ids = self.model.generate(
                question_tensor,
                max_length=max_length,
                sos_idx=self.tokenizer.word2idx[self.tokenizer.SOS_TOKEN],
                eos_idx=self.tokenizer.word2idx[self.tokenizer.EOS_TOKEN]
            )
        
        # Decodificar
        generated_ids = generated_ids[0].cpu().tolist()
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return answer
    
    def chat(self):
        """Modo interactivo de chat."""
        print("\n" + "="*50)
        print("Chatbot Universitario - Modo Interactivo")
        print("="*50)
        print("Escribe 'salir' o 'exit' para terminar\n")
        
        while True:
            try:
                question = input("Tú: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['salir', 'exit', 'quit']:
                    print("\n¡Hasta luego!")
                    break
                
                # Generar respuesta
                answer = self.predict(question)
                print(f"Bot: {answer}\n")
                
            except KeyboardInterrupt:
                print("\n\n¡Hasta luego!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Inferencia del chatbot')
    parser.add_argument('--model', type=str, default=None, help='Ruta al modelo')
    parser.add_argument('--config', type=str, default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--question', type=str, default=None, help='Pregunta única')
    
    args = parser.parse_args()
    
    # Inicializar chatbot
    chatbot = ChatBot(model_path=args.model, config_path=args.config)
    
    if args.question:
        # Modo single query
        answer = chatbot.predict(args.question)
        print(f"\nPregunta: {args.question}")
        print(f"Respuesta: {answer}")
    else:
        # Modo interactivo
        chatbot.chat()


if __name__ == "__main__":
    main()
