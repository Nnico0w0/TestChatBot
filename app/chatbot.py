"""
Lógica del chatbot para la API.
"""

import torch
import yaml
from pathlib import Path

from src.model import Seq2SeqModel
from src.tokenizer import CustomTokenizer
from src.scope_filter import ScopeFilter


class ChatBotService:
    """Servicio del chatbot para la API."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton para mantener una sola instancia del modelo."""
        if cls._instance is None:
            cls._instance = super(ChatBotService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa el servicio del chatbot."""
        if self._initialized:
            return
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = self._load_config()
        self.tokenizer = None
        self.model = None
        self.scope_filter = None
        
        self._initialized = True
    
    def _load_config(self):
        """Carga la configuración."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_model(self):
        """Carga el modelo y componentes necesarios."""
        if self.model is not None:
            return  # Ya está cargado
        
        print("Cargando tokenizer...")
        self.tokenizer = CustomTokenizer.load()
        
        print("Cargando filtro de scope...")
        self.scope_filter = ScopeFilter()
        
        print("Cargando modelo...")
        model_path = self.config['paths']['best_model_path']
        
        # Verificar si existe el modelo
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en {model_path}. "
                "Ejecuta el entrenamiento primero."
            )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = Seq2SeqModel(
            vocab_size=len(self.tokenizer),
            embedding_dim=checkpoint['config']['model']['embedding_dim'],
            hidden_dim=checkpoint['config']['model']['hidden_dim'],
            num_layers=checkpoint['config']['model']['num_layers'],
            dropout=checkpoint['config']['model']['dropout'],
            bidirectional=checkpoint['config']['model']['bidirectional'],
            use_attention=checkpoint['config']['model']['attention'],
            padding_idx=0
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Modelo cargado exitosamente!")
    
    def get_response(self, question: str, max_length: int = 50) -> dict:
        """
        Genera una respuesta para la pregunta.
        
        Args:
            question: Pregunta del usuario
            max_length: Longitud máxima de respuesta
            
        Returns:
            Diccionario con la respuesta y metadata
        """
        # Asegurar que el modelo está cargado
        if self.model is None:
            self.load_model()
        
        # Verificar scope
        is_valid, rejection_msg = self.scope_filter.filter_question(question)
        
        if not is_valid:
            return {
                'question': question,
                'answer': rejection_msg,
                'in_scope': False,
                'confidence': 0.0
            }
        
        # Preprocesar pregunta
        question_clean = question.lower().strip()
        
        try:
            # Codificar
            question_ids = self.tokenizer.encode(question_clean, add_special_tokens=True)
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
            
            return {
                'question': question,
                'answer': answer,
                'in_scope': True,
                'confidence': 1.0  # Placeholder
            }
            
        except Exception as e:
            return {
                'question': question,
                'answer': f"Error al procesar la pregunta: {str(e)}",
                'in_scope': True,
                'confidence': 0.0,
                'error': str(e)
            }
