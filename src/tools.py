from typing import Annotated, Literal
from langchain_core.tools import tool
from langgraph.types import Command
#from .app import TherapyState  # Importa el estado desde app.py

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .app import TherapyState

@tool
def end_conversation(
    state: Annotated[TherapyState, "InjectedState"],
    tool_call_id: Annotated[str, "InjectedToolCallId"],
) -> Command[Literal["__end__"]]:
    """Finaliza la conversación si no hay consentimiento"""
    messages = state.get("messages", [])
    messages.append(
        ToolMessage(
            content="Conversación finalizada por falta de consentimiento",
            tool_call_id=tool_call_id
        )
    )
    return Command(
        update={"messages": messages, "consent": False},
        goto="__end__"
    )

@tool
def transfer_to_interviewer(
    state: Annotated[TherapyState, "InjectedState"],
    tool_call_id: Annotated[str, "InjectedToolCallId"],
) -> Command[Literal["interviewer_agent"]]:
    """Inicia el cuestionario con consentimiento"""
    messages = state.get("messages", [])
    messages.extend([
        ToolMessage(
            content="Consentimiento obtenido",
            tool_call_id=tool_call_id
        ),
        HumanMessage(content="[INICIO_ENTREVISTA]")
    ])
    return Command(
        update={"messages": messages, "consent": True},
        goto="interviewer_agent"
    )
