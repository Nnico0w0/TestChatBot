"""
Capa de embeddings entrenables desde cero.
"""

import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """Capa de embeddings de palabras entrenables."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0):
        """
        Inicializa la capa de embeddings.
        
        Args:
            vocab_size: Tamaño del vocabulario
            embedding_dim: Dimensión de los vectores de embedding
            padding_idx: Índice del token de padding
        """
        super(EmbeddingLayer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Capa de embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # Inicialización de pesos
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa los pesos de la capa de embeddings."""
        # Inicialización uniforme
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # El padding debe tener embedding cero
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
    
    def forward(self, x):
        """
        Forward pass de la capa de embeddings.
        
        Args:
            x: Tensor de índices de palabras [batch_size, seq_length]
            
        Returns:
            Embeddings [batch_size, seq_length, embedding_dim]
        """
        return self.embedding(x)
