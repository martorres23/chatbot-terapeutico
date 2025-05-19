import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Carga variables de entorno
load_dotenv()

def configure_llm():
    """Configura el LLM con validación de API key"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError(
            "❌ API key no configurada. "
            "Crea un archivo .env con GOOGLE_API_KEY=tu_key_real"
        )
    
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        convert_system_message_to_human=False,
        api_key=api_key
    )
