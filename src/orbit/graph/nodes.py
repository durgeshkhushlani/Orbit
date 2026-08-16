import json
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import interrupt

from orbit.config import settings
from orbit.document_agent.generator import write_docx, write_markdown, write_pdf
from orbit.file_agent.actions import move_file, rename_file
from orbit.file_agent.scope_guard import ScopeViolation, check_path_allowed
from orbit.generation.prompt import build_document_prompt, build_prompt
from orbit.graph.state import OrbitState
from orbit.llm.ollama_client import generate
from orbit.retrieval.retriever import RetrievedChunk, retrieve

FILE_AGENT_KEYWORDS = ("rename", "move", "organize")
DOCUMENT_AGENT_KEYWORDS = (
    "save as pdf",
    "save as docx",
    "save as markdown",
    "export as pdf",
    "export as docx",
    "generate a document",
    "generate a report",
    "create a document",
    "write a report",
)

DOCUMENT_WRITERS = {"md": write_markdown, "docx": write_docx, "pdf": write_pdf}

FILE_ACTION_PROMPT = (
    "Extract a file action from the user's request as strict JSON with keys "
    '"action" ("move" or "rename"), "source" (the file path), and '
    '"destination" (the new full path for move, or just the new filename '
    "for rename). Respond with ONLY the JSON object, nothing else.\n\n"
    "Request: {query}"
)

DOCUMENT_ACTION_PROMPT = (
    "Extract a document generation request from the user's message as strict "
    'JSON with keys "format" ("md", "docx", or "pdf") and "destination" (the '
    "full output file path). Respond with ONLY the JSON object, nothing else.\n\n"
    "Request: {query}"
)


def _latest_query(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("No human message found in state")


def _is_low_confidence(chunks: list[RetrievedChunk]) -> bool:
    return not chunks or chunks[0].distance > settings.retrieval_confidence_threshold


def route_after_supervisor(state: OrbitState) -> str:
    """Picks the specialist agent for the latest query. A keyword heuristic is
    enough for now with three specialists; this is the seam to swap in an
    LLM-based classifier once more agents make keyword matching ambiguous."""
    query = _latest_query(state["messages"]).lower()
    if any(keyword in query for keyword in FILE_AGENT_KEYWORDS):
        return "file_agent"
    if any(keyword in query for keyword in DOCUMENT_AGENT_KEYWORDS):
        return "document_agent"
    return "retrieval_agent"


def supervisor_node(state: OrbitState) -> dict:
    """Entry point for routing. No state changes of its own -- routing
    happens in the conditional edge (route_after_supervisor) that follows."""
    return {}


def retrieval_node(state: OrbitState) -> dict:
    """Retrieve relevant chunks for the latest query, ground a prompt in them,
    and generate an answer. If the best match is too weak to trust, pause the
    graph via interrupt() and ask the user to clarify before answering."""
    query = _latest_query(state["messages"])
    chunks = retrieve(query)

    if _is_low_confidence(chunks):
        candidates = list(dict.fromkeys(chunk.source for chunk in chunks))
        clarification = interrupt(
            {
                "type": "clarify",
                "question": (
                    f"I'm not confident I found the right material for '{query}'. "
                    f"Closest matches were from: {', '.join(candidates) or 'nothing indexed'}. "
                    "Can you clarify or rephrase?"
                ),
            }
        )
        query = clarification
        chunks = retrieve(query)

    prompt = build_prompt(query, chunks)
    answer = generate(prompt)
    sources = list(dict.fromkeys(chunk.source for chunk in chunks))

    return {"messages": [AIMessage(content=answer)], "sources": sources}


def file_agent_node(state: OrbitState) -> dict:
    """Extract a move/rename action from the latest query, refuse outright if
    it falls outside ORBIT_ALLOWED_DIRS (the scope guardrail -- checked before
    Confirm? is ever shown, not something the user can approve past), then
    pause for human confirmation before touching disk."""
    query = _latest_query(state["messages"])

    try:
        plan = json.loads(generate(FILE_ACTION_PROMPT.format(query=query)))
        action = plan["action"]
        source = Path(plan["source"])
        destination = (
            source.with_name(plan["destination"]) if action == "rename" else Path(plan["destination"])
        )
    except (json.JSONDecodeError, KeyError):
        return {
            "messages": [
                AIMessage(
                    content="I couldn't tell exactly what file action you want -- "
                    "could you rephrase with the file name and what to do?"
                )
            ],
            "sources": [],
        }

    try:
        check_path_allowed(source)
        check_path_allowed(destination)
    except ScopeViolation as exc:
        return {"messages": [AIMessage(content=str(exc))], "sources": []}

    approved = interrupt(
        {
            "type": "confirm",
            "question": f"About to {action} '{source}' -> '{destination}'. Proceed? (yes/no)",
        }
    )

    if str(approved).strip().lower() not in ("y", "yes"):
        return {"messages": [AIMessage(content="Okay, I won't do that.")], "sources": []}

    result_path = rename_file(source, plan["destination"]) if action == "rename" else move_file(source, destination)

    return {
        "messages": [AIMessage(content=f"Done -- {action}d to {result_path}")],
        "sources": [str(result_path)],
    }


def document_agent_node(state: OrbitState) -> dict:
    """Generate a document (md/docx/pdf) grounded in retrieved context.
    The output path is scope-checked like the other agents, but per the
    plan this action is ungated -- no Confirm? step, since generating a new
    file is lower-risk than moving/deleting an existing one."""
    query = _latest_query(state["messages"])

    try:
        plan = json.loads(generate(DOCUMENT_ACTION_PROMPT.format(query=query)))
        writer = DOCUMENT_WRITERS[plan["format"]]
        destination = Path(plan["destination"])
    except (json.JSONDecodeError, KeyError):
        return {
            "messages": [
                AIMessage(
                    content="I couldn't tell what document to generate -- could you "
                    "specify the format (md/docx/pdf) and where to save it?"
                )
            ],
            "sources": [],
        }

    try:
        check_path_allowed(destination)
    except ScopeViolation as exc:
        return {"messages": [AIMessage(content=str(exc))], "sources": []}

    chunks = retrieve(query)
    content = generate(build_document_prompt(query, chunks))
    result_path = writer(content, destination)
    sources = list(dict.fromkeys(chunk.source for chunk in chunks))

    return {
        "messages": [AIMessage(content=f"Generated {plan['format']} document at {result_path}")],
        "sources": sources,
    }
