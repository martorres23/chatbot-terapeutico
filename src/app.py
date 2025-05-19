
from typing import TypedDict, Optional, Literal, List, Dict, Any, Annotated
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.graph import StateGraph, END
import json
from .config import configure_llm

llm = configure_llm() 

class TherapyState(TypedDict, total=False):
    base_questions: List[str]
    current_question_index: int
    humanized_question: str
    reformulated_question: str
    answers: Dict[str, str]
    attempt: int
    messages: Annotated[list[Any], add_messages]
    consent: Optional[bool]
    status: Literal["in_progress", "completed"]


from .constants import JSON_PATH

def get_initial_state():
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        preguntas = json.load(f).get("preguntas", [])
    
    return TherapyState(
        messages=[HumanMessage(content="Hola")],
        base_questions=preguntas,
        current_question_index=0,
        humanized_question="",
        reformulated_question="",
        attempt=0,
        status="in_progress",
        consent=None
    )

# nodo consentimiento

from typing import Annotated, Literal
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from .tools import end_conversation, transfer_to_interviewer
from .constants import CONSENT_PROMPT

consent_tools = [end_conversation, transfer_to_interviewer]

def create_consent_agent(llm):
    return create_react_agent(llm, consent_tools, CONSENT_PROMPT)

from .constants import HUMANIZE_PROMPT_TEMPLATE

# nodo humanizar preguntas
def humanize_question(state: TherapyState) -> TherapyState:
    """
    Transforma preguntas técnicas a formato empático usando LLM.

    Ejemplo:
        "Estado civil" → "¿Podría compartir cómo está conformada su situación familiar?"

    Args:
        state: Debe contener 'current_base_question'

    Returns:
        Estado con 'humanized_question' actualizada
    """

    # Verificar que el índice actual sea válido
    current_question_index = state.get("current_question_index", 0)
    base_questions = state.get("base_questions", [])

    if current_question_index >= len(base_questions):
        raise IndexError("El índice de la pregunta está fuera de rango")

    prompt = HUMANIZE_PROMPT_TEMPLATE.format(
        question=state["base_questions"][state["current_question_index"]]
    )

    humanized = llm.invoke(prompt).content

    return {
        "humanized_question": humanized,
        "attempt": 0  # Resetear contador de intentos
    }

#nodo preguntar y validar
def ask_and_validate(state: TherapyState) -> Dict:
    """
    Muestra la pregunta al paciente y valida su respuesta.

    Interacciones:
    1. Muestra pregunta humanizada mediante interrupt()
    2. Evalúa si la respuesta es clínica mente útil

    Returns:
        Dict con:
        - answer: Respuesta del paciente
        - is_valid: Bool (True si la respuesta cumple criterios clínicos)
    """

    question = state["humanized_question"] if state["attempt"] == 0 else state["reformulated_question"]

    # 1. Mostrar pregunta y capturar respuesta
    user_input = interrupt({
        "question": question,
        "attempt": state["attempt"] + 1
    })

    # 2. Validación (ejemplo básico)
    is_valid = (
        len(user_input.strip()) > 1 and
        "no quiero responder" not in user_input.lower() and
        "no sé" not in user_input.lower()
    )

    # 3. Si la respuesta es válida:
    if is_valid:
        base_question = state["base_questions"][state["current_question_index"]]
        # Actualizar el estado con la respuesta
        updates = {
            "answers": {
                **state["answers"],
                base_question: user_input
            },
            "attempt": 0,
        }

        # Verificamos si hay más preguntas
        if state["current_question_index"] < len(state["base_questions"]) - 1:
            updates["current_question_index"] = state["current_question_index"] + 1  # Avanzar índice
            return {
                **updates,
                "next_node": "humanize_question"  # --- CAMBIO 3: Usamos next_node en lugar de Command ---
            }
        else:
            return {
                **updates,
                "next_node": "end"  # Finalizar cuestionario
            }


    # 4. Si la respuesta es inválida:
    elif state["attempt"] < 2:  # Máximo 3 intentos
        state["attempt"] += 1
        return Command(goto="empathic_reformulation")  # Reformular

    else:  # Límite de intentos alcanzado
        current_q = state["humanized_question"]
        state["answers"][current_q] = "NO_RESPONSE"
        if state["current_question_index"] < len(state["base_questions"]) - 1:
            state["current_question_index"] += 1
            return Command(goto="humanize_question")  # Siguiente pregunta
        else:
            return Command(goto="end")  # Finalizar

# nodo reformular

from .constants import REFORMULATION_PROMPT_TEMPLATE

def empathic_reformulation(state: TherapyState) -> TherapyState:
    """
    Reformula la pregunta de manera más clara o empática tras una respuesta inválida,
    usando el historial de intentos y la respuesta previa del usuario.
    """
    base_question = state["base_questions"][state["current_question_index"]]  # Original
    last_humanized = state["humanized_question"]  # Versión mostrada al usuario
    user_answer = state.get("answer", "")

    reformulation_promptt = REFORMULATION_PROMPT_TEMPLATE.format(
        base_question=state["base_questions"][state["current_question_index"]],
        last_question=state["humanized_question"],
        user_answer=state.get("answer", ""),
        attempt=state["attempt"]
    )


    new_question = llm.invoke(reformulation_promptt).content

    return {
        "reformulated_question": new_question
    }

from IPython.display import display, Image
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode

def create_graph(llm):
    builder = StateGraph(TherapyState)
    
    # Nodos
    builder.add_node("consent_agent", consent_agent)
    builder.add_node("humanize_question", humanize_question)
    builder.add_node("ask_and_validate", ask_and_validate)
    builder.add_node("empathic_reformulation", empathic_reformulation)

    builder.set_entry_point("consent_agent")
    builder.add_edge("consent_agent", "humanize_question")
    builder.add_edge("humanize_question", "ask_and_validate")
    builder.add_edge("ask_and_validate", "humanize_question")
    builder.add_edge("ask_and_validate", "empathic_reformulation")  # Respuesta inválida
    builder.add_edge("empathic_reformulation", "ask_and_validate")  # Volver a preguntar
    builder.add_edge("ask_and_validate", END)

    return builder.compile()

# Uso final
graph = create_graph(llm)
