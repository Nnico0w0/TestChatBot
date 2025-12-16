#!/usr/bin/env python3
"""
Script para unificar y optimizar datasets del chatbot.

Funcionalidad:
1. Lee múltiples datasets (JSON y texto mixto)
2. Elimina duplicados de intents
3. Optimiza respuestas (30-150 palabras)
4. Valida formato JSON
5. Genera reporte de estadísticas
6. Guarda dataset unificado en data/raw/unified_dataset.json

Uso:
    python scripts/unify_datasets.py
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict


class DatasetUnifier:
    """Unifica y optimiza múltiples datasets para el chatbot."""
    
    def __init__(self, output_path: str = "data/raw/unified_dataset.json"):
        """
        Inicializa el unificador de datasets.
        
        Args:
            output_path: Ruta donde guardar el dataset unificado
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.unified_data = []
        self.stats = {
            'total_intents': 0,
            'total_questions': 0,
            'intents_with_few_questions': [],
            'duplicates_removed': 0
        }
    
    def load_json_dataset(self, path: str) -> List[Dict]:
        """
        Carga un dataset en formato JSON.
        
        Args:
            path: Ruta al archivo JSON
            
        Returns:
            Lista de intents con preguntas y respuestas
        """
        print(f"📂 Cargando JSON desde: {path}")
        
        if not Path(path).exists():
            print(f"⚠️  Archivo no encontrado: {path}")
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Normalizar estructura si es necesario
            if isinstance(data, dict):
                # Si el JSON tiene una clave raíz, extraer los intents
                for key in ['intents', 'data', 'dataset']:
                    if key in data:
                        data = data[key]
                        break
            
            if not isinstance(data, list):
                data = [data]
            
            print(f"  ✅ {len(data)} intents cargados")
            return data
            
        except json.JSONDecodeError as e:
            print(f"  ❌ Error al parsear JSON: {e}")
            return []
        except Exception as e:
            print(f"  ❌ Error inesperado: {e}")
            return []
    
    def parse_text_dataset(self, path: str) -> List[Dict]:
        """
        Parsea un dataset en formato texto mixto con JSON embebido.
        
        Args:
            path: Ruta al archivo de texto
            
        Returns:
            Lista de intents extraídos
        """
        print(f"📂 Parseando archivo de texto: {path}")
        
        if not Path(path).exists():
            print(f"⚠️  Archivo no encontrado: {path}")
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extraer bloques JSON del archivo
            json_pattern = r'\[[\s\S]*?\](?=\n\n|\Z)'
            json_blocks = re.findall(json_pattern, content)
            
            all_data = []
            for block in json_blocks:
                try:
                    data = json.loads(block)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                except json.JSONDecodeError:
                    continue
            
            print(f"  ✅ {len(all_data)} intents extraídos")
            return all_data
            
        except Exception as e:
            print(f"  ❌ Error al leer archivo: {e}")
            return []
    
    def merge_intents(self, datasets: List[List[Dict]]) -> List[Dict]:
        """
        Unifica múltiples datasets eliminando duplicados.
        
        Args:
            datasets: Lista de datasets a unificar
            
        Returns:
            Dataset unificado sin duplicados
        """
        print("\n🔄 Unificando datasets...")
        
        # Usar diccionario para eliminar duplicados por intent
        intent_map = {}
        
        for dataset in datasets:
            for item in dataset:
                if not isinstance(item, dict):
                    continue
                
                intent = item.get('intent', '')
                if not intent:
                    continue
                
                if intent in intent_map:
                    # Intent ya existe, combinar preguntas
                    existing = intent_map[intent]
                    
                    # Combinar preguntas eliminando duplicados
                    existing_questions = set(existing.get('questions', []))
                    new_questions = set(item.get('questions', []))
                    combined_questions = list(existing_questions | new_questions)
                    
                    existing['questions'] = combined_questions
                    
                    # Mantener la respuesta más larga (probablemente más completa)
                    if len(item.get('answer', '')) > len(existing.get('answer', '')):
                        existing['answer'] = item['answer']
                    
                    self.stats['duplicates_removed'] += 1
                else:
                    # Nuevo intent
                    intent_map[intent] = {
                        'intent': intent,
                        'questions': list(set(item.get('questions', []))),
                        'answer': item.get('answer', '')
                    }
        
        unified = list(intent_map.values())
        print(f"  ✅ {len(unified)} intents únicos")
        print(f"  🗑️  {self.stats['duplicates_removed']} duplicados eliminados")
        
        return unified
    
    def shorten_answer(self, text: str, max_words: int = 150) -> str:
        """
        Acorta una respuesta si excede el límite de palabras.
        
        Args:
            text: Texto a acortar
            max_words: Número máximo de palabras
            
        Returns:
            Texto acortado preservando información esencial
        """
        words = text.split()
        
        if len(words) <= max_words:
            return text
        
        # Buscar punto más cercano al límite de palabras
        shortened = ' '.join(words[:max_words])
        last_period = shortened.rfind('.')
        
        if last_period > len(shortened) * 0.7:  # Si el punto está en el último 30%
            shortened = shortened[:last_period + 1]
        else:
            # No hay punto bueno, cortar y añadir puntos suspensivos
            shortened = ' '.join(words[:max_words]) + '...'
        
        return shortened
    
    def clean_answer(self, text: str) -> str:
        """
        Limpia y optimiza una respuesta.
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto limpio y optimizado
        """
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Reemplazar [ENLACE] con placeholder más descriptivo
        text = re.sub(r'\[ENLACE\]', 'el sitio web oficial', text)
        
        # Reemplazar [EMAIL] con placeholder
        text = re.sub(r'\[EMAIL\]', 'el correo electrónico institucional', text)
        
        # Reemplazar [TELÉFONO] con placeholder
        text = re.sub(r'\[TELÉFONO\]', 'el teléfono de contacto', text)
        
        # Eliminar otros placeholders genéricos
        text = re.sub(r'\[.*?\]', '', text)
        
        # Limpiar espacios dobles resultantes
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def validate_dataset(self, data: List[Dict]) -> bool:
        """
        Valida que el dataset tenga la estructura correcta.
        
        Args:
            data: Dataset a validar
            
        Returns:
            True si el dataset es válido
        """
        print("\n✓ Validando dataset...")
        
        valid = True
        min_questions = 15
        
        for i, item in enumerate(data):
            # Validar estructura básica
            if not isinstance(item, dict):
                print(f"  ❌ Item {i} no es un diccionario")
                valid = False
                continue
            
            # Validar campos requeridos
            if 'intent' not in item:
                print(f"  ❌ Item {i} no tiene 'intent'")
                valid = False
            
            if 'questions' not in item:
                print(f"  ❌ Item {i} ({item.get('intent', 'unknown')}) no tiene 'questions'")
                valid = False
            elif not isinstance(item['questions'], list):
                print(f"  ❌ Item {i} ({item.get('intent', 'unknown')}) 'questions' no es lista")
                valid = False
            elif len(item['questions']) < min_questions:
                intent_name = item.get('intent', 'unknown')
                self.stats['intents_with_few_questions'].append({
                    'intent': intent_name,
                    'count': len(item['questions'])
                })
            
            if 'answer' not in item:
                print(f"  ❌ Item {i} ({item.get('intent', 'unknown')}) no tiene 'answer'")
                valid = False
        
        if valid:
            print("  ✅ Dataset válido")
        
        return valid
    
    def optimize_dataset(self, data: List[Dict]) -> List[Dict]:
        """
        Optimiza el dataset: acorta respuestas, elimina duplicados, etc.
        
        Args:
            data: Dataset a optimizar
            
        Returns:
            Dataset optimizado
        """
        print("\n⚙️  Optimizando dataset...")
        
        optimized = []
        
        for item in data:
            # Limpiar y acortar respuesta
            answer = self.clean_answer(item['answer'])
            answer = self.shorten_answer(answer, max_words=150)
            
            # Eliminar preguntas duplicadas
            questions = list(set(item['questions']))
            questions.sort()  # Ordenar para consistencia
            
            optimized_item = {
                'intent': item['intent'],
                'questions': questions,
                'answer': answer
            }
            
            optimized.append(optimized_item)
        
        # Ordenar por intent para mejor legibilidad
        optimized.sort(key=lambda x: x['intent'])
        
        print(f"  ✅ Dataset optimizado")
        
        return optimized
    
    def generate_report(self):
        """Genera y muestra reporte de estadísticas."""
        print("\n" + "="*60)
        print("📊 REPORTE DE UNIFICACIÓN")
        print("="*60)
        print(f"Total de intents: {self.stats['total_intents']}")
        print(f"Total de preguntas: {self.stats['total_questions']}")
        print(f"Promedio de preguntas por intent: {self.stats['total_questions'] / max(self.stats['total_intents'], 1):.1f}")
        print(f"Duplicados eliminados: {self.stats['duplicates_removed']}")
        
        if self.stats['intents_with_few_questions']:
            print(f"\n⚠️  Intents con menos de 15 preguntas:")
            for item in self.stats['intents_with_few_questions']:
                print(f"  - {item['intent']}: {item['count']} preguntas")
        
        print("="*60)
    
    def save_unified_dataset(self, data: List[Dict]):
        """
        Guarda el dataset unificado en formato JSON.
        
        Args:
            data: Dataset a guardar
        """
        print(f"\n💾 Guardando dataset en: {self.output_path}")
        
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ Dataset guardado exitosamente")
            print(f"  📁 Tamaño: {self.output_path.stat().st_size / 1024:.1f} KB")
            
        except Exception as e:
            print(f"  ❌ Error al guardar: {e}")
            sys.exit(1)
    
    def run(self, dataset_paths: List[str]):
        """
        Ejecuta el proceso completo de unificación.
        
        Args:
            dataset_paths: Lista de rutas a datasets para unificar
        """
        print("🚀 Iniciando unificación de datasets\n")
        
        # Cargar todos los datasets
        all_datasets = []
        
        for path in dataset_paths:
            if path.endswith('.json'):
                data = self.load_json_dataset(path)
            elif path.endswith('.txt'):
                data = self.parse_text_dataset(path)
            else:
                print(f"⚠️  Formato no soportado: {path}")
                continue
            
            if data:
                all_datasets.append(data)
        
        if not all_datasets:
            print("❌ No se pudieron cargar datasets")
            sys.exit(1)
        
        # Unificar datasets
        unified = self.merge_intents(all_datasets)
        
        # Optimizar
        optimized = self.optimize_dataset(unified)
        
        # Validar
        if not self.validate_dataset(optimized):
            print("\n⚠️  Dataset tiene problemas, pero se guardará de todas formas")
        
        # Actualizar estadísticas
        self.stats['total_intents'] = len(optimized)
        self.stats['total_questions'] = sum(len(item['questions']) for item in optimized)
        
        # Guardar
        self.save_unified_dataset(optimized)
        
        # Generar reporte
        self.generate_report()
        
        print("\n✅ Proceso completado exitosamente!")


def main():
    """Función principal."""
    # Definir rutas de datasets a unificar
    dataset_paths = [
        "data/raw/qa_dataset.txt",
        "datasets/curza_dataset.json",  # Se agregará cuando esté disponible
    ]
    
    # Crear unificador y ejecutar
    unifier = DatasetUnifier(output_path="data/raw/unified_dataset.json")
    unifier.run(dataset_paths)


if __name__ == "__main__":
    main()
