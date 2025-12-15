"""
Filtro de relevancia para rechazar preguntas fuera del dominio.
"""

import json
import random
import yaml
from typing import List
import torch
import torch.nn.functional as F


class ScopeFilter:
    """Filtro para validar si una pregunta está dentro del scope del chatbot."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el filtro de scope.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.keywords_file = self.config['scope_filter']['keywords_file']
        self.threshold = self.config['scope_filter']['similarity_threshold']
        
        # Cargar keywords y mensajes de rechazo
        self._load_keywords()
    
    def _load_keywords(self):
        """Carga las keywords del dominio y mensajes de rechazo."""
        with open(self.keywords_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.keywords = set(data['keywords'])
        self.rejection_messages = data['rejection_messages']
    
    def is_in_scope(self, text: str) -> bool:
        """
        Verifica si el texto está dentro del scope del chatbot.
        
        Args:
            text: Texto a verificar
            
        Returns:
            True si está en scope, False si no
        """
        # Tokenizar y convertir a minúsculas
        tokens = text.lower().split()
        
        # Contar cuántas keywords aparecen
        matches = 0
        for token in tokens:
            if token in self.keywords:
                matches += 1
        
        # Calcular ratio de coincidencia
        if len(tokens) == 0:
            return False
        
        match_ratio = matches / len(tokens)
        
        # Si al menos una keyword coincide o ratio supera threshold
        return matches > 0 or match_ratio >= self.threshold
    
    def get_rejection_message(self) -> str:
        """
        Retorna un mensaje de rechazo aleatorio.
        
        Returns:
            Mensaje de rechazo
        """
        return random.choice(self.rejection_messages)
    
    def filter_question(self, text: str) -> tuple:
        """
        Filtra una pregunta y retorna resultado.
        
        Args:
            text: Pregunta a filtrar
            
        Returns:
            (is_valid, message): Tupla con validez y mensaje
        """
        if self.is_in_scope(text):
            return True, None
        else:
            return False, self.get_rejection_message()


class AdvancedScopeFilter(ScopeFilter):
    """Filtro avanzado con embeddings de palabras."""
    
    def __init__(self, config_path: str = "config.yaml", tokenizer=None):
        """
        Inicializa el filtro avanzado.
        
        Args:
            config_path: Ruta al archivo de configuración
            tokenizer: Tokenizador del modelo (opcional)
        """
        super().__init__(config_path)
        self.tokenizer = tokenizer
    
    def calculate_semantic_similarity(self, text: str, keyword_embeddings) -> float:
        """
        Calcula similitud semántica usando embeddings.
        
        Args:
            text: Texto a evaluar
            keyword_embeddings: Embeddings de keywords
            
        Returns:
            Score de similitud
        """
        if self.tokenizer is None:
            return 0.0
        
        # Tokenizar texto
        tokens = self.tokenizer.tokenize(text.lower())
        
        # Calcular embeddings promedio (esto requeriría el modelo cargado)
        # Por ahora, usar el método simple
        return 0.0
    
    def is_in_scope_advanced(self, text: str) -> bool:
        """
        Versión avanzada del filtro con embeddings.
        
        Args:
            text: Texto a verificar
            
        Returns:
            True si está en scope, False si no
        """
        # Primero intentar método simple
        if self.is_in_scope(text):
            return True
        
        # Si hay tokenizer disponible, usar método avanzado
        if self.tokenizer is not None:
            # Aquí se podría implementar comparación con embeddings
            # de keywords del dominio
            pass
        
        return False
