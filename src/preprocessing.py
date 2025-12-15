"""
Preprocesamiento de datos para el chatbot.
Limpia y prepara el dataset para entrenamiento.
"""

import json
import pickle
import re
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
import nltk
from sklearn.model_selection import train_test_split

# Descargar recursos de NLTK si no están disponibles
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class DataPreprocessor:
    """Preprocesador de datos para el chatbot."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Inicializa el preprocesador con la configuración."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = self.config['data']['raw_data_path']
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        self.train_split = self.config['data']['train_split']
        
        # Crear directorio de salida si no existe
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """
        Limpia el texto removiendo caracteres especiales y normalizando.
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remover [ENLACE], [EMAIL], etc.
        text = re.sub(r'\[.*?\]', '', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def parse_dataset(self, file_path: str) -> List[Dict]:
        """
        Parsea el archivo de dataset en formato texto/JSON mixto.
        
        Args:
            file_path: Ruta al archivo del dataset
            
        Returns:
            Lista de diccionarios con intents, preguntas y respuestas
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extraer bloques JSON del archivo
        json_pattern = r'\[[\s\S]*?\](?=\n\n|\Z)'
        json_blocks = re.findall(json_pattern, content)
        
        all_data = []
        for block in json_blocks:
            try:
                # Intentar parsear cada bloque como JSON
                data = json.loads(block)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
            except json.JSONDecodeError:
                continue
        
        # Aplanar estructuras anidadas
        flattened_data = []
        for item in all_data:
            if isinstance(item, dict):
                # Manejar estructuras con claves anidadas
                if 'carreras' in item:
                    flattened_data.extend(item['carreras'])
                elif 'horarios_catedras' in item:
                    flattened_data.extend(item['horarios_catedras'])
                elif 'intents' in item:
                    flattened_data.extend(item['intents'])
                else:
                    flattened_data.append(item)
        
        return flattened_data
    
    def create_qa_pairs(self, data: List[Dict]) -> List[Tuple[str, str]]:
        """
        Crea pares de pregunta-respuesta desde los datos parseados.
        
        Args:
            data: Lista de diccionarios con intents
            
        Returns:
            Lista de tuplas (pregunta, respuesta)
        """
        qa_pairs = []
        
        for item in data:
            if 'questions' in item and 'answer' in item:
                answer = self.clean_text(item['answer'])
                
                for question in item['questions']:
                    question_clean = self.clean_text(question)
                    qa_pairs.append((question_clean, answer))
        
        return qa_pairs
    
    def process(self):
        """Procesa el dataset completo y guarda los datos procesados."""
        print("Iniciando preprocesamiento...")
        
        # Parsear dataset
        print(f"Leyendo dataset desde: {self.raw_data_path}")
        data = self.parse_dataset(self.raw_data_path)
        print(f"Intents encontrados: {len(data)}")
        
        # Crear pares de pregunta-respuesta
        qa_pairs = self.create_qa_pairs(data)
        print(f"Pares de Q&A creados: {len(qa_pairs)}")
        
        # Split train/val
        train_pairs, val_pairs = train_test_split(
            qa_pairs, 
            train_size=self.train_split,
            random_state=42
        )
        
        print(f"Datos de entrenamiento: {len(train_pairs)}")
        print(f"Datos de validación: {len(val_pairs)}")
        
        # Guardar datos procesados
        train_path = self.processed_data_path / "train.pkl"
        val_path = self.processed_data_path / "val.pkl"
        
        with open(train_path, 'wb') as f:
            pickle.dump(train_pairs, f)
        
        with open(val_path, 'wb') as f:
            pickle.dump(val_pairs, f)
        
        print(f"Datos guardados en: {self.processed_data_path}")
        print("Preprocesamiento completado exitosamente!")
        
        return train_pairs, val_pairs


def main():
    """Función principal para ejecutar el preprocesamiento."""
    preprocessor = DataPreprocessor()
    preprocessor.process()


if __name__ == "__main__":
    main()
