import json
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import interrupt

from orbit.config import settings
from orbit.document_agent.generator import write_docx, write_markdown, write_pdf
from orbit.email_agent.mailer import send_email
from orbit.file_agent.actions import move_file, rename_file
from orbit.file_agent.scope_guard import ScopeViolation, check_path_allowed
from orbit.generation.prompt import build_document_prompt, build_prompt
from orbit.graph.state import OrbitState
from orbit.ingestion.indexer import index_file
from orbit.llm.ollama_client import generate
from orbit.retrieval.retriever import RetrievedChunk, retrieve
from orbit.web_agent.extract import extract_content
from orbit.web_agent.search import search_web

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
EMAIL_AGENT_KEYWORDS = ("email", "e-mail", "send mail", "send an email")
WEB_AGENT_KEYWORDS = (
    "search the web",
    "search online",
    "search google",
    "look up online",
    "web search",
    "google it",
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

WEB_ACTION_PROMPT = (
    "Extract a web search request from the user's message as strict JSON with keys "
    '"query" (the search query text, with instruction phrases like "search the web '
    'for" stripped out), "save_as" ("pdf", "docx", "md", or null if the user just '
    'wants an answer, not a saved file), and "destination" (the full output file '
    "path, required only if save_as is not null, otherwise null). Respond with ONLY "
    "the JSON object, nothing else.\n\n"
    "Request: {query}"
)

EMAIL_ACTION_PROMPT = (
    "Extract an email to send from the user's request as strict JSON with keys "
    '"to" (recipient address -- if the user says "myself"/"me", use {user_email}), '
    '"subject", "body", and "attachment" (an absolute file path if the user wants '
    "a file attached, matched against the indexed sources below, or null). "
    "Respond with ONLY the JSON object, nothing else.\n\n"
    "Indexed sources:\n{sources}\n\n"
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
    enough for now with five specialists; this is the seam to swap in an
    LLM-based classifier once more agents make keyword matching ambiguous.

    Web Agent is checked before Document Agent: a request like "search the
    web for X and save as pdf" contains a document-style "save as pdf" phrase
    but is fundamentally a web request, not a local-context document one.
    """
    query = _latest_query(state["messages"]).lower()
    if any(keyword in query for keyword in FILE_AGENT_KEYWORDS):
        return "file_agent"
    if any(keyword in query for keyword in WEB_AGENT_KEYWORDS):
        return "web_agent"
    if any(keyword in query for keyword in DOCUMENT_AGENT_KEYWORDS):
        return "document_agent"
    if any(keyword in query for keyword in EMAIL_AGENT_KEYWORDS):
        return "email_agent"
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


def email_agent_node(state: OrbitState) -> dict:
    """Extract a send-email action from the latest query -- retrieving indexed
    sources first so the LLM can resolve a referenced document (e.g. "email me
    my resume") to an actual file path -- refuse outright if any attachment
    falls outside ORBIT_ALLOWED_DIRS, then pause for human confirmation before
    sending anything."""
    query = _latest_query(state["messages"])
    chunks = retrieve(query)
    sources = list(dict.fromkeys(chunk.source for chunk in chunks))

    try:
        prompt = EMAIL_ACTION_PROMPT.format(
            user_email=settings.orbit_user_email or "(no default address configured)",
            sources="\n".join(sources) or "(nothing indexed)",
            query=query,
        )
        plan = json.loads(generate(prompt))
        to = plan["to"]
        subject = plan["subject"]
        body = plan["body"]
        attachment = plan.get("attachment") or None
    except (json.JSONDecodeError, KeyError):
        return {
            "messages": [
                AIMessage(
                    content="I couldn't tell exactly what email you want sent -- "
                    "could you specify the recipient, subject, and body?"
                )
            ],
            "sources": [],
        }

    attachment_path = None
    if attachment:
        try:
            attachment_path = check_path_allowed(Path(attachment))
        except ScopeViolation as exc:
            return {"messages": [AIMessage(content=str(exc))], "sources": []}

    approved = interrupt(
        {
            "type": "confirm",
            "question": (
                f"About to email '{subject}' to {to}"
                f"{f' with attachment {attachment_path.name}' if attachment_path else ''}. "
                "Proceed? (yes/no)"
            ),
        }
    )

    if str(approved).strip().lower() not in ("y", "yes"):
        return {"messages": [AIMessage(content="Okay, I won't send that.")], "sources": []}

    send_email(to, subject, body, attachment_path)

    return {
        "messages": [AIMessage(content=f"Sent -- emailed '{subject}' to {to}")],
        "sources": [str(attachment_path)] if attachment_path else [],
    }


def web_agent_node(state: OrbitState) -> dict:
    """Search the web and answer grounded in the extracted page content --
    read-only, no Confirm? gate. If the request also asks to save the results
    to a file, that path is scope-checked and gated the same way as File/
    Email/Document actions, and the saved file is auto-indexed into Chroma
    afterwards so it's immediately retrievable."""
    query = _latest_query(state["messages"])

    try:
        plan = json.loads(generate(WEB_ACTION_PROMPT.format(query=query)))
        search_query = plan["query"]
        save_as = plan.get("save_as") or None
        destination = plan.get("destination") or None
    except (json.JSONDecodeError, KeyError):
        return {
            "messages": [AIMessage(content="I couldn't tell what to search for -- could you rephrase?")],
            "sources": [],
        }

    if save_as and not destination:
        return {
            "messages": [
                AIMessage(content="Where should I save the results? Please give me a destination path.")
            ],
            "sources": [],
        }

    results = search_web(search_query)
    if not results:
        return {
            "messages": [AIMessage(content=f"I couldn't find anything on the web for '{search_query}'.")],
            "sources": [],
        }

    web_chunks = [
        RetrievedChunk(text=extract_content(r.url) or r.snippet, source=r.url, distance=0.0)
        for r in results
    ]

    if save_as is None:
        answer = generate(build_prompt(search_query, web_chunks))
        return {
            "messages": [AIMessage(content=answer)],
            "sources": [chunk.source for chunk in web_chunks],
        }

    destination_path = Path(destination)
    try:
        check_path_allowed(destination_path)
    except ScopeViolation as exc:
        return {"messages": [AIMessage(content=str(exc))], "sources": []}

    approved = interrupt(
        {
            "type": "confirm",
            "question": (
                f"About to search the web for '{search_query}' and save the results as "
                f"{save_as} to '{destination_path}'. Proceed? (yes/no)"
            ),
        }
    )

    if str(approved).strip().lower() not in ("y", "yes"):
        return {"messages": [AIMessage(content="Okay, I won't do that.")], "sources": []}

    content = generate(build_document_prompt(search_query, web_chunks))
    result_path = DOCUMENT_WRITERS[save_as](content, destination_path)
    index_summary = index_file(result_path)

    return {
        "messages": [
            AIMessage(
                content=(
                    f"Saved web research on '{search_query}' as {save_as} to {result_path} "
                    f"and indexed it ({index_summary['chunks_indexed']} chunks)."
                )
            )
        ],
        "sources": [str(result_path)] + [chunk.source for chunk in web_chunks],
    }
