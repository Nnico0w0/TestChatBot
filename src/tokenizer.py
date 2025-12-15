"""
Tokenizador custom para el chatbot.
Construye y gestiona el vocabulario del dominio.
"""

import pickle
import yaml
from pathlib import Path
from typing import List, Dict
from collections import Counter
import nltk

# Descargar recursos de NLTK si no están disponibles
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


class CustomTokenizer:
    """Tokenizador custom con vocabulario específico del dominio."""
    
    # Tokens especiales
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    SOS_TOKEN = '<SOS>'  # Start of sequence
    EOS_TOKEN = '<EOS>'  # End of sequence
    
    def __init__(self, config_path: str = "config.yaml"):
        """Inicializa el tokenizador."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.min_word_freq = self.config['data']['min_word_freq']
        self.vocab_path = Path(self.config['data']['vocab_path'])
        self.tokenizer_path = Path(self.config['paths']['tokenizer_dir']) / "tokenizer.pkl"
        
        # Vocabularios
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # Inicializar con tokens especiales
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Inicializa los tokens especiales en el vocabulario."""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN
        ]
        
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza el texto en palabras.
        
        Args:
            text: Texto a tokenizar
            
        Returns:
            Lista de tokens
        """
        # Usar word_tokenize de NLTK
        tokens = nltk.word_tokenize(text, language='spanish')
        return tokens
    
    def build_vocab(self, texts: List[str]):
        """
        Construye el vocabulario desde una lista de textos.
        
        Args:
            texts: Lista de textos para construir el vocabulario
        """
        print("Construyendo vocabulario...")
        
        # Contar frecuencias de palabras
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)
        
        print(f"Total de palabras únicas: {len(self.word_freq)}")
        
        # Agregar palabras que cumplan con la frecuencia mínima
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_word_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"Tamaño del vocabulario: {len(self.word2idx)}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Codifica texto a índices.
        
        Args:
            text: Texto a codificar
            add_special_tokens: Si agregar SOS y EOS tokens
            
        Returns:
            Lista de índices
        """
        tokens = self.tokenize(text)
        
        # Convertir tokens a índices
        indices = []
        if add_special_tokens:
            indices.append(self.word2idx[self.SOS_TOKEN])
        
        for token in tokens:
            idx = self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])
            indices.append(idx)
        
        if add_special_tokens:
            indices.append(self.word2idx[self.EOS_TOKEN])
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodifica índices a texto.
        
        Args:
            indices: Lista de índices
            skip_special_tokens: Si omitir tokens especiales
            
        Returns:
            Texto decodificado
        """
        special_tokens = {
            self.word2idx[self.PAD_TOKEN],
            self.word2idx[self.SOS_TOKEN],
            self.word2idx[self.EOS_TOKEN]
        }
        
        tokens = []
        for idx in indices:
            if skip_special_tokens and idx in special_tokens:
                continue
            
            token = self.idx2word.get(idx, self.UNK_TOKEN)
            tokens.append(token)
        
        return ' '.join(tokens)
    
    def save(self):
        """Guarda el tokenizador."""
        # Crear directorios si no existen
        self.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar tokenizador completo
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(self, f)
        
        # Guardar vocabulario en JSON para inspección
        import json
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()},
            'vocab_size': len(self.word2idx)
        }
        
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"Tokenizador guardado en: {self.tokenizer_path}")
        print(f"Vocabulario guardado en: {self.vocab_path}")
    
    @classmethod
    def load(cls, tokenizer_path: str = None):
        """
        Carga un tokenizador guardado.
        
        Args:
            tokenizer_path: Ruta al tokenizador
            
        Returns:
            Tokenizador cargado
        """
        if tokenizer_path is None:
            # Usar ruta por defecto
            with open("config.yaml", 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            tokenizer_path = Path(config['paths']['tokenizer_dir']) / "tokenizer.pkl"
        
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        return tokenizer
    
    def __len__(self):
        """Retorna el tamaño del vocabulario."""
        return len(self.word2idx)


def main():
    """Función principal para construir y guardar el tokenizador."""
    import pickle
    
    # Cargar datos procesados
    with open("data/processed/train.pkl", 'rb') as f:
        train_pairs = pickle.load(f)
    
    with open("data/processed/val.pkl", 'rb') as f:
        val_pairs = pickle.load(f)
    
    # Extraer todos los textos (preguntas y respuestas)
    all_texts = []
    for question, answer in train_pairs + val_pairs:
        all_texts.append(question)
        all_texts.append(answer)
    
    # Construir tokenizador
    tokenizer = CustomTokenizer()
    tokenizer.build_vocab(all_texts)
    
    # Guardar
    tokenizer.save()
    
    print("\nTokenizador construido exitosamente!")


if __name__ == "__main__":
    main()
