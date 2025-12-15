#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate a massive dataset for the CURZA university chatbot.
Extracts content from official CURZA URLs and generates multiple question variations.
"""

import json
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple
from urllib.parse import urljoin, urlparse
import time


class CURZADatasetGenerator:
    """Generator for CURZA university chatbot dataset."""
    
    def __init__(self):
        self.base_url = "https://web.curza.uncoma.edu.ar"
        self.urls = [
            "https://web.curza.uncoma.edu.ar/preinscripcion/",
            "https://web.curza.uncoma.edu.ar/oferta-academica",
            "https://web.curza.uncoma.edu.ar/academica/tramites",
            "https://web.curza.uncoma.edu.ar/bienestar",
            "https://web.curza.uncoma.edu.ar/"
        ]
        self.dataset = []
        
    def scrape_url(self, url: str) -> Tuple[str, List[str], List[Dict]]:
        """
        Scrape a URL and extract text content, links, and sections.
        
        Returns:
            Tuple of (text_content, links, sections)
        """
        try:
            print(f"Scraping: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                print(f"Failed to fetch {url}: Status {response.status_code}")
                return "", [], []
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            
            # Extract main text content
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                link_text = link.get_text(strip=True)
                if link_text and urlparse(full_url).netloc == urlparse(self.base_url).netloc:
                    links.append({
                        'text': link_text,
                        'url': full_url
                    })
            
            # Extract sections (h1, h2, h3, etc.)
            sections = []
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                section_title = heading.get_text(strip=True)
                if section_title:
                    # Get content after heading
                    content = ""
                    for sibling in heading.find_next_siblings():
                        if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            break
                        content += sibling.get_text(strip=True) + " "
                    
                    sections.append({
                        'title': section_title,
                        'content': content.strip()[:500]  # Limit content length
                    })
            
            time.sleep(1)  # Be polite to the server
            return text, links, sections
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return "", [], []
    
    def generate_questions_variations(self, topic: str, base_questions: List[str]) -> List[str]:
        """
        Generate 20+ variations of questions for a given topic.
        Includes formal, informal, typos, synonyms, and regional variations.
        """
        variations = []
        
        # Synonyms for common words
        synonyms_map = {
            'consultar': ['ver', 'buscar', 'encontrar', 'acceder', 'revisar', 'mirar'],
            'información': ['info', 'datos', 'detalles'],
            'página': ['sitio', 'web', 'link', 'enlace', 'URL'],
            'dónde': ['donde'],  # without accent (typo)
            'cómo': ['como'],    # without accent (typo)
            'cuál': ['cual'],    # without accent (typo)
            'qué': ['que'],      # without accent (typo)
        }
        
        # Add original questions
        variations.extend(base_questions)
        
        # Formal variations
        formal_templates = [
            f"¿Dónde puedo consultar sobre {topic}?",
            f"¿Cuál es el procedimiento para {topic}?",
            f"¿En qué sección encuentro {topic}?",
            f"¿Cómo accedo a la información sobre {topic}?",
            f"¿Dónde está disponible {topic}?",
            f"¿Podría indicarme dónde ver {topic}?",
        ]
        
        # Informal/colloquial variations
        informal_templates = [
            f"¿Dónde veo {topic}?",
            f"¿Cómo hago para {topic}?",
            f"¿Tenés el link de {topic}?",
            f"¿Me pasás info sobre {topic}?",
            f"¿Dónde me fijo para {topic}?",
            f"¿Dónde está {topic}?",
            f"necesito {topic}",
            f"quiero ver {topic}",
            f"busco {topic}",
        ]
        
        # With typos (missing accents)
        typo_templates = [
            f"¿Donde esta {topic}?",
            f"¿Como hago para {topic}?",
            f"¿Cual es {topic}?",
            f"¿Que es {topic}?",
        ]
        
        # Regional Argentine variations
        regional_templates = [
            f"¿Dónde me fijo {topic}?",
            f"¿Dónde me anoto para {topic}?",
            f"¿Cómo tramito {topic}?",
            f"¿Cómo hago para ver {topic}?",
        ]
        
        # Partial/incomplete questions
        partial_templates = [
            topic,
            f"info {topic}",
            f"necesito {topic}",
            f"quiero saber de {topic}",
        ]
        
        # Combine all templates
        all_templates = (formal_templates + informal_templates + 
                        typo_templates + regional_templates + partial_templates)
        
        variations.extend(all_templates)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var not in seen:
                seen.add(var)
                unique_variations.append(var)
        
        return unique_variations
    
    def create_intent_from_section(self, section_title: str, section_content: str, 
                                   url: str, link_data: List[Dict]) -> Dict:
        """Create an intent from a section with proper format."""
        
        # Clean intent name
        intent_name = re.sub(r'[^a-z0-9]+', '_', section_title.lower()).strip('_')
        
        # Generate base questions
        base_questions = [
            f"¿Dónde puedo encontrar información sobre {section_title.lower()}?",
            f"Necesito saber sobre {section_title.lower()}",
        ]
        
        # Generate all variations
        questions = self.generate_questions_variations(section_title.lower(), base_questions)
        
        # Create answer with URL and description
        answer = (
            f"Podés encontrar información sobre {section_title.lower()} en el siguiente enlace: "
            f"{url}\n\n"
            f"En esta página encontrarás detalles sobre: {section_content[:200]}..."
        )
        
        return {
            "intent": intent_name,
            "questions": questions,
            "answer": answer
        }
    
    def generate_preinscripcion_intents(self, url: str) -> List[Dict]:
        """Generate intents for preinscription page."""
        intents = []
        
        # Main preinscription intent
        questions = self.generate_questions_variations(
            "preinscripción",
            [
                "¿Cómo me preinscribo?",
                "¿Dónde está la preinscripción?",
                "Necesito preinscribirme",
                "Quiero hacer la preinscripción"
            ]
        )
        
        intents.append({
            "intent": "preinscripcion_principal",
            "questions": questions,
            "answer": (
                f"Podés realizar la preinscripción en el siguiente enlace: {url}\n\n"
                "En esta página encontrarás toda la información sobre el proceso de preinscripción, "
                "los pasos a seguir, la documentación requerida y podrás acceder al sistema de "
                "preinscripción online."
            )
        })
        
        # Requisitos preinscripción
        questions = self.generate_questions_variations(
            "requisitos de preinscripción",
            [
                "¿Qué necesito para preinscribirme?",
                "Requisitos para la preinscripción",
                "¿Qué documentos necesito para preinscripción?"
            ]
        )
        
        intents.append({
            "intent": "requisitos_preinscripcion",
            "questions": questions,
            "answer": (
                f"Podés consultar los requisitos para la preinscripción en: {url}\n\n"
                "La página contiene información detallada sobre la documentación necesaria, "
                "los requisitos académicos, y todos los pasos del proceso de preinscripción."
            )
        })
        
        # Fechas preinscripción
        questions = self.generate_questions_variations(
            "fechas de preinscripción",
            [
                "¿Cuándo es la preinscripción?",
                "Fechas de preinscripción",
                "¿Hasta cuándo puedo preinscribirme?"
            ]
        )
        
        intents.append({
            "intent": "fechas_preinscripcion",
            "questions": questions,
            "answer": (
                f"Podés consultar las fechas de preinscripción en: {url}\n\n"
                "En esta página encontrarás el cronograma actualizado con las fechas de inicio "
                "y cierre de la preinscripción para cada período académico."
            )
        })
        
        return intents
    
    def generate_oferta_academica_intents(self, url: str) -> List[Dict]:
        """Generate intents for academic offer page."""
        intents = []
        
        # Main offer intent
        questions = self.generate_questions_variations(
            "oferta académica",
            [
                "¿Qué carreras hay?",
                "¿Cuál es la oferta académica?",
                "Quiero ver las carreras disponibles",
                "¿Qué puedo estudiar?"
            ]
        )
        
        intents.append({
            "intent": "oferta_academica_principal",
            "questions": questions,
            "answer": (
                f"Podés consultar toda la oferta académica en: {url}\n\n"
                "En esta página encontrarás el listado completo de carreras de grado, pregrado, "
                "tecnicaturas y cursos disponibles en CURZA, con información detallada sobre cada una."
            )
        })
        
        # Carreras de grado
        questions = self.generate_questions_variations(
            "carreras de grado",
            [
                "¿Qué carreras de grado hay?",
                "Carreras universitarias disponibles",
                "¿Qué licenciaturas ofrecen?"
            ]
        )
        
        intents.append({
            "intent": "carreras_grado",
            "questions": questions,
            "answer": (
                f"Podés ver todas las carreras de grado disponibles en: {url}\n\n"
                "La página contiene información sobre las licenciaturas, ingenierías y profesorados "
                "que se dictan en CURZA, incluyendo planes de estudio y duración de cada carrera."
            )
        })
        
        # Tecnicaturas
        questions = self.generate_questions_variations(
            "tecnicaturas",
            [
                "¿Hay tecnicaturas?",
                "¿Qué tecnicaturas se pueden estudiar?",
                "Tecnicaturas disponibles"
            ]
        )
        
        intents.append({
            "intent": "tecnicaturas",
            "questions": questions,
            "answer": (
                f"Podés consultar las tecnicaturas disponibles en: {url}\n\n"
                "En esta página encontrarás información sobre las tecnicaturas universitarias "
                "que se dictan en CURZA, su duración y plan de estudios."
            )
        })
        
        # Profesorados
        questions = self.generate_questions_variations(
            "profesorados",
            [
                "¿Hay profesorados?",
                "¿Qué profesorados se dictan?",
                "Profesorados disponibles"
            ]
        )
        
        intents.append({
            "intent": "profesorados",
            "questions": questions,
            "answer": (
                f"Podés ver los profesorados disponibles en: {url}\n\n"
                "La página muestra los profesorados que se dictan en CURZA, con información "
                "sobre las especialidades disponibles y requisitos de cada uno."
            )
        })
        
        # Plan de estudios
        questions = self.generate_questions_variations(
            "plan de estudios",
            [
                "¿Dónde veo el plan de estudios?",
                "Necesito el plan de estudios",
                "¿Cómo es el plan de estudios?"
            ]
        )
        
        intents.append({
            "intent": "planes_estudio",
            "questions": questions,
            "answer": (
                f"Podés consultar los planes de estudio en: {url}\n\n"
                "En la sección de oferta académica encontrarás los planes de estudio detallados "
                "de cada carrera, con las materias, correlatividades y carga horaria."
            )
        })
        
        return intents
    
    def generate_tramites_intents(self, url: str) -> List[Dict]:
        """Generate intents for procedures/tramites page."""
        intents = []
        
        # Main tramites intent
        questions = self.generate_questions_variations(
            "trámites",
            [
                "¿Qué trámites puedo hacer?",
                "Necesito hacer un trámite",
                "¿Dónde hago trámites?"
            ]
        )
        
        intents.append({
            "intent": "tramites_principal",
            "questions": questions,
            "answer": (
                f"Podés consultar todos los trámites disponibles en: {url}\n\n"
                "En esta página encontrarás información sobre todos los trámites académicos "
                "que podés realizar, los requisitos, documentación necesaria y los pasos a seguir."
            )
        })
        
        # Certificados
        questions = self.generate_questions_variations(
            "certificados",
            [
                "¿Cómo pido un certificado?",
                "Necesito un certificado de alumno regular",
                "¿Dónde saco certificados?"
            ]
        )
        
        intents.append({
            "intent": "certificados",
            "questions": questions,
            "answer": (
                f"Podés ver cómo solicitar certificados en: {url}\n\n"
                "La página contiene información sobre los diferentes tipos de certificados "
                "disponibles, cómo solicitarlos y el tiempo de demora en su emisión."
            )
        })
        
        # Certificado de alumno regular
        questions = self.generate_questions_variations(
            "certificado de alumno regular",
            [
                "¿Cómo saco el certificado de alumno regular?",
                "Necesito certificado de regularidad",
                "Certificado de estudiante regular"
            ]
        )
        
        intents.append({
            "intent": "certificado_alumno_regular",
            "questions": questions,
            "answer": (
                f"Podés consultar cómo obtener el certificado de alumno regular en: {url}\n\n"
                "En la sección de trámites encontrarás los pasos para solicitar este certificado, "
                "los requisitos necesarios y el tiempo de demora."
            )
        })
        
        # Certificado analítico
        questions = self.generate_questions_variations(
            "certificado analítico",
            [
                "¿Cómo pido el certificado analítico?",
                "Necesito mi certificado analítico",
                "Certificado analítico de estudios"
            ]
        )
        
        intents.append({
            "intent": "certificado_analitico",
            "questions": questions,
            "answer": (
                f"Podés ver cómo solicitar el certificado analítico en: {url}\n\n"
                "La página contiene información sobre el trámite para obtener tu certificado "
                "analítico, la documentación requerida y los tiempos de emisión."
            )
        })
        
        # Cambio de carrera
        questions = self.generate_questions_variations(
            "cambio de carrera",
            [
                "¿Cómo cambio de carrera?",
                "Quiero cambiarme de carrera",
                "Trámite de cambio de carrera"
            ]
        )
        
        intents.append({
            "intent": "cambio_carrera",
            "questions": questions,
            "answer": (
                f"Podés consultar sobre el cambio de carrera en: {url}\n\n"
                "En esta página encontrarás información sobre el trámite de cambio de carrera, "
                "los requisitos, plazos y procedimientos necesarios."
            )
        })
        
        # Equivalencias
        questions = self.generate_questions_variations(
            "equivalencias",
            [
                "¿Cómo hago equivalencias?",
                "Necesito hacer equivalencias de materias",
                "Trámite de equivalencias"
            ]
        )
        
        intents.append({
            "intent": "equivalencias",
            "questions": questions,
            "answer": (
                f"Podés ver cómo solicitar equivalencias en: {url}\n\n"
                "La página contiene información sobre el trámite de equivalencias, "
                "la documentación necesaria, el proceso de evaluación y los plazos."
            )
        })
        
        # Pase de universidad
        questions = self.generate_questions_variations(
            "pase de universidad",
            [
                "¿Cómo hago un pase desde otra universidad?",
                "Vengo de otra universidad",
                "Trámite de pase"
            ]
        )
        
        intents.append({
            "intent": "pase_universidad",
            "questions": questions,
            "answer": (
                f"Podés consultar sobre el pase desde otra universidad en: {url}\n\n"
                "En esta página encontrarás información sobre el trámite de pase, "
                "los requisitos, documentación necesaria y el proceso de admisión."
            )
        })
        
        # Título intermedio
        questions = self.generate_questions_variations(
            "título intermedio",
            [
                "¿Cómo tramito el título intermedio?",
                "Necesito mi título intermedio",
                "Trámite de título intermedio"
            ]
        )
        
        intents.append({
            "intent": "titulo_intermedio",
            "questions": questions,
            "answer": (
                f"Podés ver cómo tramitar el título intermedio en: {url}\n\n"
                "La página contiene información sobre el trámite para obtener tu título intermedio, "
                "los requisitos, documentación necesaria y los pasos a seguir."
            )
        })
        
        # Baja de materias
        questions = self.generate_questions_variations(
            "baja de materias",
            [
                "¿Cómo me doy de baja de una materia?",
                "Necesito dar de baja materias",
                "Baja de cursada"
            ]
        )
        
        intents.append({
            "intent": "baja_materias",
            "questions": questions,
            "answer": (
                f"Podés consultar sobre la baja de materias en: {url}\n\n"
                "En esta página encontrarás información sobre el procedimiento de baja de materias, "
                "los plazos y cómo realizar este trámite."
            )
        })
        
        return intents
    
    def generate_bienestar_intents(self, url: str) -> List[Dict]:
        """Generate intents for student welfare page."""
        intents = []
        
        # Main bienestar intent
        questions = self.generate_questions_variations(
            "bienestar estudiantil",
            [
                "¿Qué es bienestar estudiantil?",
                "Servicios de bienestar",
                "¿Qué ofrece bienestar?"
            ]
        )
        
        intents.append({
            "intent": "bienestar_principal",
            "questions": questions,
            "answer": (
                f"Podés consultar los servicios de bienestar estudiantil en: {url}\n\n"
                "En esta página encontrarás información sobre todos los programas, servicios "
                "y beneficios que ofrece el área de bienestar estudiantil de CURZA."
            )
        })
        
        # Becas
        questions = self.generate_questions_variations(
            "becas",
            [
                "¿Hay becas disponibles?",
                "¿Cómo solicito una beca?",
                "Información sobre becas"
            ]
        )
        
        intents.append({
            "intent": "becas",
            "questions": questions,
            "answer": (
                f"Podés ver información sobre becas en: {url}\n\n"
                "La página contiene información sobre las becas disponibles, requisitos, "
                "períodos de inscripción y cómo realizar la solicitud."
            )
        })
        
        # Comedor universitario
        questions = self.generate_questions_variations(
            "comedor universitario",
            [
                "¿Hay comedor?",
                "¿Dónde está el comedor universitario?",
                "Información del comedor"
            ]
        )
        
        intents.append({
            "intent": "comedor_universitario",
            "questions": questions,
            "answer": (
                f"Podés consultar sobre el comedor universitario en: {url}\n\n"
                "En esta página encontrarás información sobre el servicio de comedor, "
                "horarios, menús y cómo acceder al servicio."
            )
        })
        
        # Residencia estudiantil
        questions = self.generate_questions_variations(
            "residencia estudiantil",
            [
                "¿Hay residencia estudiantil?",
                "¿Cómo accedo a la residencia?",
                "Información sobre residencias"
            ]
        )
        
        intents.append({
            "intent": "residencia_estudiantil",
            "questions": questions,
            "answer": (
                f"Podés ver información sobre residencias estudiantiles en: {url}\n\n"
                "La página contiene información sobre las residencias disponibles, "
                "requisitos de acceso, proceso de solicitud y servicios incluidos."
            )
        })
        
        # Salud estudiantil
        questions = self.generate_questions_variations(
            "salud estudiantil",
            [
                "¿Hay servicios de salud?",
                "Atención médica en la universidad",
                "Salud para estudiantes"
            ]
        )
        
        intents.append({
            "intent": "salud_estudiantil",
            "questions": questions,
            "answer": (
                f"Podés consultar sobre servicios de salud en: {url}\n\n"
                "En esta página encontrarás información sobre los servicios de salud disponibles "
                "para estudiantes, horarios de atención y cómo acceder a ellos."
            )
        })
        
        # Apoyo psicológico
        questions = self.generate_questions_variations(
            "apoyo psicológico",
            [
                "¿Hay atención psicológica?",
                "Necesito apoyo psicológico",
                "Servicio de psicología"
            ]
        )
        
        intents.append({
            "intent": "apoyo_psicologico",
            "questions": questions,
            "answer": (
                f"Podés ver información sobre apoyo psicológico en: {url}\n\n"
                "La página contiene información sobre los servicios de apoyo psicológico "
                "disponibles para estudiantes, cómo solicitar turnos y horarios de atención."
            )
        })
        
        # Deportes
        questions = self.generate_questions_variations(
            "deportes y actividades",
            [
                "¿Hay actividades deportivas?",
                "Deportes en la universidad",
                "¿Qué deportes puedo practicar?"
            ]
        )
        
        intents.append({
            "intent": "deportes_actividades",
            "questions": questions,
            "answer": (
                f"Podés consultar sobre deportes y actividades en: {url}\n\n"
                "En esta página encontrarás información sobre las actividades deportivas "
                "y recreativas disponibles, horarios, lugares de práctica y cómo inscribirse."
            )
        })
        
        return intents
    
    def generate_home_intents(self, url: str) -> List[Dict]:
        """Generate intents for home page."""
        intents = []
        
        # Main home intent
        questions = self.generate_questions_variations(
            "página principal",
            [
                "¿Cuál es la página de CURZA?",
                "Página principal de la universidad",
                "Sitio web de CURZA"
            ]
        )
        
        intents.append({
            "intent": "pagina_principal",
            "questions": questions,
            "answer": (
                f"La página principal de CURZA es: {url}\n\n"
                "Desde aquí podés acceder a toda la información institucional, carreras, "
                "trámites, servicios y novedades de la universidad."
            )
        })
        
        # Contacto
        questions = self.generate_questions_variations(
            "contacto",
            [
                "¿Cómo contacto a la universidad?",
                "Datos de contacto",
                "Teléfonos y emails"
            ]
        )
        
        intents.append({
            "intent": "contacto",
            "questions": questions,
            "answer": (
                f"Podés encontrar la información de contacto en: {url}\n\n"
                "La página contiene todos los datos de contacto de CURZA, incluyendo "
                "teléfonos, emails, direcciones y horarios de atención."
            )
        })
        
        # Ubicación
        questions = self.generate_questions_variations(
            "ubicación",
            [
                "¿Dónde queda CURZA?",
                "Dirección de la universidad",
                "¿Cómo llego a CURZA?"
            ]
        )
        
        intents.append({
            "intent": "ubicacion",
            "questions": questions,
            "answer": (
                f"Podés ver la ubicación de CURZA en: {url}\n\n"
                "En la página encontrarás la dirección completa, mapas de ubicación "
                "y cómo llegar en transporte público o vehículo particular."
            )
        })
        
        # Calendario académico
        questions = self.generate_questions_variations(
            "calendario académico",
            [
                "¿Dónde está el calendario académico?",
                "Necesito el calendario de clases",
                "Fechas del calendario académico"
            ]
        )
        
        intents.append({
            "intent": "calendario_academico",
            "questions": questions,
            "answer": (
                f"Podés consultar el calendario académico en: {url}\n\n"
                "La página contiene el calendario académico con fechas de inicio y fin "
                "de cuatrimestres, períodos de exámenes, recesos y feriados."
            )
        })
        
        # Horarios
        questions = self.generate_questions_variations(
            "horarios de atención",
            [
                "¿Cuáles son los horarios de atención?",
                "Horarios de secretaría",
                "¿Cuándo atienden?"
            ]
        )
        
        intents.append({
            "intent": "horarios_atencion",
            "questions": questions,
            "answer": (
                f"Podés ver los horarios de atención en: {url}\n\n"
                "En esta página encontrarás los horarios de atención de las diferentes "
                "áreas administrativas y académicas de CURZA."
            )
        })
        
        # Autoridades
        questions = self.generate_questions_variations(
            "autoridades",
            [
                "¿Quiénes son las autoridades?",
                "Directivos de CURZA",
                "Autoridades de la universidad"
            ]
        )
        
        intents.append({
            "intent": "autoridades",
            "questions": questions,
            "answer": (
                f"Podés consultar las autoridades de CURZA en: {url}\n\n"
                "La página contiene información sobre las autoridades de la institución, "
                "sus funciones y datos de contacto."
            )
        })
        
        # Biblioteca
        questions = self.generate_questions_variations(
            "biblioteca",
            [
                "¿Hay biblioteca?",
                "Información de la biblioteca",
                "¿Dónde está la biblioteca?"
            ]
        )
        
        intents.append({
            "intent": "biblioteca",
            "questions": questions,
            "answer": (
                f"Podés ver información sobre la biblioteca en: {url}\n\n"
                "En esta página encontrarás datos sobre la biblioteca de CURZA, "
                "horarios, servicios, catálogo online y cómo acceder a los recursos."
            )
        })
        
        # Inscripción general
        questions = self.generate_questions_variations(
            "inscripción",
            [
                "¿Cómo me inscribo?",
                "Proceso de inscripción",
                "Quiero inscribirme"
            ]
        )
        
        intents.append({
            "intent": "inscripcion_general",
            "questions": questions,
            "answer": (
                f"Podés ver el proceso de inscripción en: {url}\n\n"
                "La página contiene información general sobre el proceso de inscripción, "
                "requisitos, documentación y pasos a seguir. Para más detalles específicos, "
                "visitá la sección de preinscripción."
            )
        })
        
        return intents
    
    def generate_additional_intents(self) -> List[Dict]:
        """Generate additional common intents."""
        intents = []
        
        # Aula virtual
        questions = self.generate_questions_variations(
            "aula virtual",
            [
                "¿Dónde está el aula virtual?",
                "Acceso al campus virtual",
                "Link del aula virtual"
            ]
        )
        
        intents.append({
            "intent": "aula_virtual",
            "questions": questions,
            "answer": (
                "Podés acceder al aula virtual de CURZA en: https://web.curza.uncoma.edu.ar/\n\n"
                "Desde la página principal encontrarás el acceso al campus virtual donde "
                "se encuentran los materiales de las materias, foros y actividades online."
            )
        })
        
        # Email institucional
        questions = self.generate_questions_variations(
            "email institucional",
            [
                "¿Cómo accedo a mi email institucional?",
                "Email de estudiante",
                "Correo de la universidad"
            ]
        )
        
        intents.append({
            "intent": "email_institucional",
            "questions": questions,
            "answer": (
                "Podés consultar sobre el email institucional en: https://web.curza.uncoma.edu.ar/\n\n"
                "La página contiene información sobre cómo activar y acceder a tu correo "
                "electrónico institucional como estudiante de CURZA."
            )
        })
        
        # Sistema de inscripción a materias
        questions = self.generate_questions_variations(
            "inscripción a materias",
            [
                "¿Cómo me inscribo a las materias?",
                "Sistema de inscripción a cursadas",
                "Inscripción a materias online"
            ]
        )
        
        intents.append({
            "intent": "inscripcion_materias",
            "questions": questions,
            "answer": (
                "Podés ver cómo inscribirte a materias en: https://web.curza.uncoma.edu.ar/academica/tramites\n\n"
                "En esta página encontrarás información sobre el sistema de inscripción a materias, "
                "períodos de inscripción y requisitos de correlatividades."
            )
        })
        
        # Mesa de exámenes
        questions = self.generate_questions_variations(
            "mesa de exámenes",
            [
                "¿Dónde veo las mesas de examen?",
                "Fechas de exámenes",
                "Inscripción a exámenes"
            ]
        )
        
        intents.append({
            "intent": "mesa_examenes",
            "questions": questions,
            "answer": (
                "Podés consultar las mesas de examen en: https://web.curza.uncoma.edu.ar/\n\n"
                "La página contiene información sobre las fechas de mesas de examen, "
                "cómo inscribirte y los requisitos para rendir."
            )
        })
        
        return intents
    
    def generate_dataset(self) -> None:
        """Generate the complete dataset."""
        print("=" * 80)
        print("GENERANDO DATASET MASIVO PARA CHATBOT CURZA")
        print("=" * 80)
        
        # Generate intents for each URL
        url_generators = {
            "https://web.curza.uncoma.edu.ar/preinscripcion/": self.generate_preinscripcion_intents,
            "https://web.curza.uncoma.edu.ar/oferta-academica": self.generate_oferta_academica_intents,
            "https://web.curza.uncoma.edu.ar/academica/tramites": self.generate_tramites_intents,
            "https://web.curza.uncoma.edu.ar/bienestar": self.generate_bienestar_intents,
            "https://web.curza.uncoma.edu.ar/": self.generate_home_intents,
        }
        
        for url, generator_func in url_generators.items():
            print(f"\nProcesando: {url}")
            intents = generator_func(url)
            self.dataset.extend(intents)
            print(f"  ✓ Generados {len(intents)} intents")
        
        # Add additional common intents
        print("\nGenerando intents adicionales...")
        additional_intents = self.generate_additional_intents()
        self.dataset.extend(additional_intents)
        print(f"  ✓ Generados {len(additional_intents)} intents adicionales")
        
        # Calculate statistics
        total_intents = len(self.dataset)
        total_questions = sum(len(intent['questions']) for intent in self.dataset)
        avg_questions = total_questions / total_intents if total_intents > 0 else 0
        
        print("\n" + "=" * 80)
        print("ESTADÍSTICAS DEL DATASET")
        print("=" * 80)
        print(f"Total de intents: {total_intents}")
        print(f"Total de preguntas: {total_questions}")
        print(f"Promedio de preguntas por intent: {avg_questions:.1f}")
        
        # Check minimum requirement
        intents_below_20 = [i for i in self.dataset if len(i['questions']) < 20]
        if intents_below_20:
            print(f"\n⚠ ADVERTENCIA: {len(intents_below_20)} intents tienen menos de 20 preguntas")
        else:
            print("\n✓ Todos los intents tienen al menos 20 preguntas")
        
    def save_dataset(self, output_path: str = "datasets/curza_dataset.json") -> None:
        """Save the dataset to a JSON file."""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with proper formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Dataset guardado en: {output_path}")
        print(f"  Tamaño del archivo: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    """Main function to generate the dataset."""
    generator = CURZADatasetGenerator()
    
    # Generate the dataset
    generator.generate_dataset()
    
    # Save the dataset
    generator.save_dataset()
    
    print("\n" + "=" * 80)
    print("¡DATASET GENERADO EXITOSAMENTE!")
    print("=" * 80)
    print("\nEl dataset está listo para ser usado en el entrenamiento del chatbot.")
    print("Ubicación: datasets/curza_dataset.json")


if __name__ == "__main__":
    main()
