
from langchain_core.prompts import ChatPromptTemplate
import os
from pathlib import Path

# Ruta ABSOLUTA desde la raíz del proyecto
BASE_DIR = Path(__file__).parent.parent  # Sube a /chatbot-terapeutico/
JSON_PATH = str(BASE_DIR / 'static' / 'Evaluacion_Inicial.json')

# Verificación automática
if not Path(JSON_PATH).exists():
    raise FileNotFoundError(f"""
    ❌ Error crítico: Archivo no encontrado
    Ruta probada: {JSON_PATH}
    Directorio actual: {os.getcwd()}
    ¿Existe static/? {os.path.exists(BASE_DIR / 'static')}
    Archivos en static/: {os.listdir(BASE_DIR / 'static') if os.path.exists(BASE_DIR / 'static') else 'No existe'}
    """)


# 1. Prompt para consentimiento (no necesita variables externas)
CONSENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Eres el psicólogo Élia. Debes:
1. Presentarte brevemente.
2. Explicar qué es una evaluación psicológica inicial.
3. Pedir consentimiento para continuar con la entrevista.
4. Manejar respuestas:
   - SI/afirmativo: Te despides e Informas que iniciará la entrevista (Invocar transfer_to_interviewer).
   - NO/negativo: Explicar cortésmente e Invocar end_conversation.
   - Respuestas ambiguas: Pedir confirmación.

Mantén un tono profesional pero cercano. Usa emojis moderadamente."""),
    ("placeholder", "{messages}")
])

# 2. Plantillas para reformulación (usan format strings)
HUMANIZE_PROMPT_TEMPLATE = """
Como psicólogo experto, reformule esta pregunta para hacerla más empática.

Pregunta original: {question}

Directrices:
1. Use lenguaje cálido y de apoyo
2. Demuestre comprensión emocional
3. Mantenga el objetivo clínico
4. Conserve el contexto original
5. Solo una versión reformulada

Pregunta reformulada:"""

REFORMULATION_PROMPT_TEMPLATE = """
Como terapeuta, reformule esta pregunta considerando:

- Objetivo clínico: "{base_question}"
- Última versión: "{last_question}"
- Respuesta del usuario: "{user_answer}"
- Intentos previos: {attempt}

Reformulación empática:"""
