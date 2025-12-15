"""
Script de entrenamiento del modelo Seq2Seq.
Incluye checkpoints, early stopping y logging.
"""

import argparse
import pickle
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.model import Seq2SeqModel
from src.tokenizer import CustomTokenizer


class QADataset(Dataset):
    """Dataset para pares de pregunta-respuesta."""
    
    def __init__(self, data, tokenizer, max_length):
        """
        Inicializa el dataset.
        
        Args:
            data: Lista de tuplas (pregunta, respuesta)
            tokenizer: Tokenizador
            max_length: Longitud máxima de secuencias
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        question, answer = self.data[idx]
        
        # Codificar pregunta y respuesta
        question_ids = self.tokenizer.encode(question, add_special_tokens=True)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=True)
        
        # Truncar si es necesario
        if len(question_ids) > self.max_length:
            question_ids = question_ids[:self.max_length]
        if len(answer_ids) > self.max_length:
            answer_ids = answer_ids[:self.max_length]
        
        return {
            'question': torch.tensor(question_ids, dtype=torch.long),
            'answer': torch.tensor(answer_ids, dtype=torch.long),
            'question_len': len(question_ids),
            'answer_len': len(answer_ids)
        }


def collate_fn(batch):
    """Función para crear batches con padding."""
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    question_lens = [item['question_len'] for item in batch]
    answer_lens = [item['answer_len'] for item in batch]
    
    # Padding
    questions_padded = nn.utils.rnn.pad_sequence(questions, batch_first=True, padding_value=0)
    answers_padded = nn.utils.rnn.pad_sequence(answers, batch_first=True, padding_value=0)
    
    return {
        'question': questions_padded,
        'answer': answers_padded,
        'question_len': torch.tensor(question_lens, dtype=torch.long),
        'answer_len': torch.tensor(answer_lens, dtype=torch.long)
    }


class Trainer:
    """Clase para entrenar el modelo."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el trainer.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Configuración
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Crear directorios
        Path(self.config['paths']['checkpoints_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['paths']['final_model_dir']).mkdir(parents=True, exist_ok=True)
        
        # Cargar tokenizer
        print("Cargando tokenizer...")
        self.tokenizer = CustomTokenizer.load()
        
        # Cargar datos
        print("Cargando datos...")
        self.train_data = self._load_data('data/processed/train.pkl')
        self.val_data = self._load_data('data/processed/val.pkl')
        
        # Crear datasets y dataloaders
        self.train_dataset = QADataset(
            self.train_data,
            self.tokenizer,
            self.config['data']['max_seq_length']
        )
        self.val_dataset = QADataset(
            self.val_data,
            self.tokenizer,
            self.config['data']['max_seq_length']
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Crear modelo
        print("Creando modelo...")
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # Optimizador y criterio
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignorar padding
        
        # Variables de entrenamiento
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _load_data(self, path):
        """Carga datos desde pickle."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _create_model(self):
        """Crea el modelo Seq2Seq."""
        return Seq2SeqModel(
            vocab_size=len(self.tokenizer),
            embedding_dim=self.config['model']['embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            bidirectional=self.config['model']['bidirectional'],
            use_attention=self.config['model']['attention'],
            padding_idx=0
        )
    
    def train_epoch(self):
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Época {self.current_epoch + 1}")
        
        for batch in progress_bar:
            # Mover a device
            questions = batch['question'].to(self.device)
            answers = batch['answer'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(questions, answers, teacher_forcing_ratio=0.5)
            
            # Calcular loss
            # Reshape para CrossEntropyLoss
            outputs = outputs[:, 1:].contiguous().view(-1, outputs.shape[-1])
            answers = answers[:, 1:].contiguous().view(-1)
            
            loss = self.criterion(outputs, answers)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Valida el modelo."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                questions = batch['question'].to(self.device)
                answers = batch['answer'].to(self.device)
                
                outputs = self.model(questions, answers, teacher_forcing_ratio=0)
                
                outputs = outputs[:, 1:].contiguous().view(-1, outputs.shape[-1])
                answers = answers[:, 1:].contiguous().view(-1)
                
                loss = self.criterion(outputs, answers)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Guarda un checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Guardar checkpoint periódico
        if epoch % self.config['training']['checkpoint_every'] == 0:
            checkpoint_path = Path(self.config['paths']['checkpoints_dir']) / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint guardado: {checkpoint_path}")
        
        # Guardar mejor modelo
        if is_best:
            best_path = Path(self.config['paths']['best_model_path'])
            torch.save(checkpoint, best_path)
            print(f"Mejor modelo guardado: {best_path}")
    
    def train(self, num_epochs=None):
        """
        Entrena el modelo.
        
        Args:
            num_epochs: Número de épocas (usa config si es None)
        """
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        print(f"\nIniciando entrenamiento por {num_epochs} épocas...")
        print(f"Tamaño del vocabulario: {len(self.tokenizer)}")
        print(f"Datos de entrenamiento: {len(self.train_data)}")
        print(f"Datos de validación: {len(self.val_data)}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Entrenar
            train_loss = self.train_epoch()
            
            # Validar
            val_loss = self.validate()
            
            # Calcular perplexity
            train_perplexity = np.exp(train_loss)
            val_perplexity = np.exp(val_loss)
            
            print(f"\nÉpoca {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Perplexity: {train_perplexity:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.4f}")
            
            # Guardar checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch + 1, val_loss, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping en época {epoch + 1}")
                break
        
        print("\n¡Entrenamiento completado!")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Entrenar el chatbot')
    parser.add_argument('--config', type=str, default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--epochs', type=int, default=None, help='Número de épocas')
    
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
