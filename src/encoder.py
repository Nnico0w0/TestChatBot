"""
LSTM Encoder Bidireccional para el chatbot.
"""

import torch
import torch.nn as nn
from src.embeddings import EmbeddingLayer


class LSTMEncoder(nn.Module):
    """Encoder LSTM Bidireccional para procesar la pregunta."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        padding_idx: int = 0
    ):
        """
        Inicializa el encoder.
        
        Args:
            vocab_size: Tamaño del vocabulario
            embedding_dim: Dimensión de embeddings
            hidden_dim: Dimensión oculta del LSTM
            num_layers: Número de capas LSTM
            dropout: Tasa de dropout
            bidirectional: Si usar LSTM bidireccional
            padding_idx: Índice del token de padding
        """
        super(LSTMEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Capa de embeddings
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim, padding_idx)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths=None):
        """
        Forward pass del encoder.
        
        Args:
            x: Tensor de índices [batch_size, seq_length]
            lengths: Longitudes de las secuencias (opcional)
            
        Returns:
            outputs: Salidas del LSTM [batch_size, seq_length, hidden_dim * num_directions]
            hidden: Estado oculto final [num_layers * num_directions, batch_size, hidden_dim]
            cell: Estado de celda final [num_layers * num_directions, batch_size, hidden_dim]
        """
        # Embeddings
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Empaquetar secuencias si se proporcionan longitudes
        if lengths is not None:
            # Ordenar por longitud descendente
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            embedded_sorted = embedded[sorted_idx]
            
            # Empaquetar
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded_sorted,
                lengths_sorted.cpu(),
                batch_first=True,
                enforce_sorted=True
            )
            
            # LSTM
            packed_output, (hidden, cell) = self.lstm(packed)
            
            # Desempaquetar
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True
            )
            
            # Revertir el ordenamiento
            _, unsorted_idx = sorted_idx.sort()
            outputs = outputs[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
            cell = cell[:, unsorted_idx, :]
        else:
            # Sin empaquetar
            outputs, (hidden, cell) = self.lstm(embedded)
        
        return outputs, hidden, cell
