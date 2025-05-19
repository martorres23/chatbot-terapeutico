from .app import create_graph, get_initial_state
from .constants import JSON_PATH  # Añade esto a constants.py
from langchain_google_genai import ChatGoogleGenerativeAI

# Configura LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Inicializa el grafo
graph = create_graph(llm)
initial_state = get_initial_state(JSON_PATH)

# Ejecuta el flujo
for step in graph.stream(initial_state):
    print(f"Estado actual: {step}")
