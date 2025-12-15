"""
LSTM Decoder con mecanismo de Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.embeddings import EmbeddingLayer


class Attention(nn.Module):
    """Mecanismo de Attention para el decoder."""
    
    def __init__(self, hidden_dim: int):
        """
        Inicializa el mecanismo de attention.
        
        Args:
            hidden_dim: Dimensión oculta
        """
        super(Attention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Capa para calcular scores de attention
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        """
        Calcula los pesos de attention y el contexto.
        
        Args:
            hidden: Estado oculto del decoder [batch_size, hidden_dim]
            encoder_outputs: Salidas del encoder [batch_size, src_len, hidden_dim]
            
        Returns:
            context: Vector de contexto [batch_size, hidden_dim]
            attention_weights: Pesos de attention [batch_size, src_len]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repetir hidden para cada posición del encoder
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch_size, src_len, hidden_dim]
        
        # Concatenar hidden con encoder_outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, hidden_dim]
        
        # Calcular scores
        attention_scores = self.v(energy).squeeze(2)
        # attention_scores: [batch_size, src_len]
        
        # Calcular pesos con softmax
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Calcular contexto como suma ponderada
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # context: [batch_size, 1, hidden_dim]
        
        context = context.squeeze(1)
        # context: [batch_size, hidden_dim]
        
        return context, attention_weights


class LSTMDecoder(nn.Module):
    """Decoder LSTM con Attention para generar respuestas."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
        padding_idx: int = 0
    ):
        """
        Inicializa el decoder.
        
        Args:
            vocab_size: Tamaño del vocabulario
            embedding_dim: Dimensión de embeddings
            hidden_dim: Dimensión oculta del LSTM
            num_layers: Número de capas LSTM
            dropout: Tasa de dropout
            use_attention: Si usar mecanismo de attention
            padding_idx: Índice del token de padding
        """
        super(LSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Capa de embeddings
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim, padding_idx)
        
        # LSTM
        lstm_input_dim = embedding_dim + hidden_dim if use_attention else embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention
        if use_attention:
            self.attention = Attention(hidden_dim)
        
        # Capa de salida
        output_dim = hidden_dim * 2 if use_attention else hidden_dim
        self.fc_out = nn.Linear(output_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        """
        Forward pass del decoder para un paso temporal.
        
        Args:
            x: Input token [batch_size]
            hidden: Estado oculto [num_layers, batch_size, hidden_dim]
            cell: Estado de celda [num_layers, batch_size, hidden_dim]
            encoder_outputs: Salidas del encoder [batch_size, src_len, hidden_dim]
            
        Returns:
            output: Logits de salida [batch_size, vocab_size]
            hidden: Nuevo estado oculto
            cell: Nuevo estado de celda
            attention_weights: Pesos de attention (si se usa)
        """
        # x: [batch_size] -> [batch_size, 1]
        x = x.unsqueeze(1)
        
        # Embeddings
        embedded = self.embedding(x)  # [batch_size, 1, embedding_dim]
        embedded = self.dropout(embedded)
        
        attention_weights = None
        
        if self.use_attention:
            # Calcular attention usando el último hidden state
            last_hidden = hidden[-1]  # [batch_size, hidden_dim]
            context, attention_weights = self.attention(last_hidden, encoder_outputs)
            
            # Concatenar embedding con contexto
            lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
            # lstm_input: [batch_size, 1, embedding_dim + hidden_dim]
        else:
            lstm_input = embedded
        
        # LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: [batch_size, 1, hidden_dim]
        
        # Preparar output para la capa de salida
        if self.use_attention:
            output = torch.cat((output.squeeze(1), context), dim=1)
            # output: [batch_size, hidden_dim * 2]
        else:
            output = output.squeeze(1)
            # output: [batch_size, hidden_dim]
        
        # Capa de salida
        prediction = self.fc_out(output)
        # prediction: [batch_size, vocab_size]
        
        return prediction, hidden, cell, attention_weights
