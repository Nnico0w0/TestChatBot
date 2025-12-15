"""
Modelo completo Seq2Seq con Encoder-Decoder y Attention.
"""

import torch
import torch.nn as nn
from src.encoder import LSTMEncoder
from src.decoder import LSTMDecoder


class Seq2SeqModel(nn.Module):
    """Modelo completo Seq2Seq para el chatbot."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
        padding_idx: int = 0
    ):
        """
        Inicializa el modelo Seq2Seq.
        
        Args:
            vocab_size: Tamaño del vocabulario
            embedding_dim: Dimensión de embeddings
            hidden_dim: Dimensión oculta
            num_layers: Número de capas LSTM
            dropout: Tasa de dropout
            bidirectional: Si usar encoder bidireccional
            use_attention: Si usar mecanismo de attention
            padding_idx: Índice del token de padding
        """
        super(Seq2SeqModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Encoder
        self.encoder = LSTMEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            padding_idx=padding_idx
        )
        
        # Decoder (nota: encoder bidireccional produce hidden_dim * 2)
        decoder_hidden_dim = hidden_dim * self.num_directions
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention,
            padding_idx=padding_idx
        )
        
        # Proyección para adaptar estados del encoder bidireccional al decoder
        if bidirectional:
            self.hidden_projection = nn.Linear(hidden_dim * 2, hidden_dim * 2)
            self.cell_projection = nn.Linear(hidden_dim * 2, hidden_dim * 2)
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass del modelo.
        
        Args:
            src: Secuencias fuente [batch_size, src_len]
            trg: Secuencias objetivo [batch_size, trg_len]
            teacher_forcing_ratio: Probabilidad de usar teacher forcing
            
        Returns:
            outputs: Predicciones [batch_size, trg_len, vocab_size]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        
        # Tensor para almacenar salidas del decoder
        outputs = torch.zeros(batch_size, trg_len, self.vocab_size).to(src.device)
        
        # Encoder
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Si el encoder es bidireccional, combinar estados
        if self.bidirectional:
            # hidden: [num_layers * 2, batch_size, hidden_dim]
            # Necesitamos: [num_layers, batch_size, hidden_dim * 2]
            
            # Reorganizar hidden
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
            
            # Reorganizar cell
            cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
            
            # Proyectar a la dimensión correcta
            hidden = torch.tanh(self.hidden_projection(hidden))
            cell = torch.tanh(self.cell_projection(cell))
        
        # Primer input del decoder es el token <SOS>
        input_token = trg[:, 0]
        
        # Decoder paso a paso
        for t in range(1, trg_len):
            # Decoder forward
            output, hidden, cell, _ = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs
            )
            
            # Almacenar predicción
            outputs[:, t, :] = output
            
            # Decidir si usar teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            # Siguiente input
            if use_teacher_forcing:
                input_token = trg[:, t]
            else:
                input_token = output.argmax(1)
        
        return outputs
    
    def generate(self, src, max_length=50, sos_idx=2, eos_idx=3):
        """
        Genera una secuencia de salida dado un input.
        
        Args:
            src: Secuencia fuente [batch_size, src_len]
            max_length: Longitud máxima de la secuencia generada
            sos_idx: Índice del token <SOS>
            eos_idx: Índice del token <EOS>
            
        Returns:
            generated: Secuencia generada [batch_size, gen_len]
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # Encoder
            encoder_outputs, hidden, cell = self.encoder(src)
            
            # Si el encoder es bidireccional, combinar estados
            if self.bidirectional:
                hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
                hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
                
                cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
                cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
                
                hidden = torch.tanh(self.hidden_projection(hidden))
                cell = torch.tanh(self.cell_projection(cell))
            
            # Inicializar con token <SOS>
            input_token = torch.tensor([sos_idx] * batch_size).to(src.device)
            
            generated = [input_token.unsqueeze(1)]
            
            # Generar tokens uno por uno
            for _ in range(max_length):
                output, hidden, cell, _ = self.decoder(
                    input_token,
                    hidden,
                    cell,
                    encoder_outputs
                )
                
                # Obtener token con mayor probabilidad
                predicted = output.argmax(1)
                generated.append(predicted.unsqueeze(1))
                
                # Usar predicción como siguiente input
                input_token = predicted
                
                # Detener si todos generaron <EOS>
                if (predicted == eos_idx).all():
                    break
            
            # Concatenar tokens generados
            generated = torch.cat(generated, dim=1)
            
        return generated
