import os
from pathlib import Path
from typing import Literal, TypedDict, Annotated
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors as genai_errors
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from pydantic import BaseModel
import chromadb
from langgraph.graph import StateGraph, END
import operator

load_dotenv()

EMBEDDINGS_DIR = Path("data/embeddings")

gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
chroma_client = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))
collection = chroma_client.get_or_create_collection(name="contracts")


@retry(
    retry=retry_if_exception(lambda e: isinstance(e, genai_errors.ClientError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=10, max=60),
    reraise=True,
)
def _llm_call(**kwargs):
    return gemini.models.generate_content(**kwargs)


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    intent: str                          # "qa" | "risk_scan" | "draft_memo"
    override_intent: str                 # bypasses classifier when set
    source_filter: str                   # restrict retrieval to one contract filename
    retrieved_chunks: list[dict]
    answer: str
    risk_flags: list[dict]
    memo: str
    sources: list[str]
    needs_human_review: bool


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    result = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


def retrieve(query: str, n_results: int = 5, source_filter: str = "") -> list[dict]:
    """Retrieve top-n relevant chunks from ChromaDB."""
    if collection.count() == 0:
        return []
    query_embedding = get_embedding(query)
    kwargs: dict = dict(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if source_filter:
        kwargs["where"] = {"source": {"$eq": source_filter}}
    results = collection.query(**kwargs)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "relevance": round(1 - dist, 3)
        })
    return chunks


def format_context(chunks: list[dict]) -> str:
    """Format chunks into a readable context block for the prompt."""
    parts = []
    for i, c in enumerate(chunks):
        parts.append(
            f"[Source {i+1}: {c['source']}, Page {c['page']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def format_sources(chunks: list[dict]) -> list[str]:
    """Deduplicated list of source citations."""
    seen = set()
    sources = []
    for c in chunks:
        ref = f"{c['source']} (p.{c['page']})"
        if ref not in seen:
            seen.add(ref)
            sources.append(ref)
    return sources


# ── Node 1: Intent Classifier ──────────────────────────────────────────────────

def classify_intent(state: AgentState) -> AgentState:
    """Classify user query into one of three intents."""
    if state.get("override_intent"):
        print(f"[Intent] {state['override_intent']} (overridden)")
        return {**state, "intent": state["override_intent"]}

    prompt = f"""Classify the following procurement query into exactly one category.

Categories:
- qa: User wants to find specific information or get an answer from the contract
- risk_scan: User wants to identify risks, red flags, or problematic clauses
- draft_memo: User wants a summary or approval memo drafted

Query: {state['query']}

Reply with just one word: qa, risk_scan, or draft_memo"""

    response = _llm_call(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    intent = response.text.strip().lower()
    if intent not in ["qa", "risk_scan", "draft_memo"]:
        intent = "qa"
    print(f"[Intent] {intent}")
    return {**state, "intent": intent}


# ── Node 2: RAG Retriever ──────────────────────────────────────────────────────

def rag_retriever(state: AgentState) -> AgentState:
    """Retrieve relevant chunks based on the query."""
    n = 8 if state["intent"] == "risk_scan" else 5
    chunks = retrieve(state["query"], n_results=n, source_filter=state.get("source_filter", ""))
    print(f"[Retriever] Found {len(chunks)} chunks")
    return {**state, "retrieved_chunks": chunks}


# ── Node 3a: QA Synthesiser ────────────────────────────────────────────────────

def qa_synthesiser(state: AgentState) -> AgentState:
    """Answer a specific question using retrieved context."""
    context = format_context(state["retrieved_chunks"])
    prompt = f"""You are a procurement intelligence assistant. Answer the question below using ONLY the contract excerpts provided.

Rules:
- Be specific and direct
- Always cite the source and page number for every claim (e.g. "per contract_01.pdf, Page 3")
- If the answer is not in the context, say "This information was not found in the provided contract sections"
- Keep the answer under 200 words

Contract excerpts:
{context}

Question: {state['query']}

Answer:"""

    response = _llm_call(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    sources = format_sources(state["retrieved_chunks"])
    print(f"[QA] Answer generated")
    return {**state, "answer": response.text.strip(), "sources": sources}


# ── Node 3b: Risk Scanner ──────────────────────────────────────────────────────

class RiskFlag(BaseModel):
    risk_type: str
    severity: Literal["high", "medium", "low"]
    clause_summary: str
    source: str
    page: int
    recommendation: str


RISK_TAXONOMY = [
    "unlimited_liability",
    "inadequate_liability_cap",
    "auto_renewal_trap",
    "ip_ownership_risk",
    "missing_sla",
    "data_lock_in",
    "payment_penalty",
    "forced_upgrade_risk",
    "unilateral_change_right",
    "short_claims_window"
]

def risk_scanner(state: AgentState) -> AgentState:
    """Scan retrieved chunks for compliance risks and return structured flags."""
    context = format_context(state["retrieved_chunks"])
    prompt = f"""You are a procurement risk analyst. Review the contract excerpts below and identify risk clauses.

Risk types to look for: {', '.join(RISK_TAXONOMY)}

For each risk found, populate: risk_type (from the list above), severity (high/medium/low),
clause_summary (one sentence on the problematic clause), source (filename), page (page number),
and recommendation (one sentence on what to negotiate or flag).

Return an empty list if no risks are found.

Contract excerpts:
{context}"""

    response = _llm_call(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[RiskFlag],
        ),
    )

    flags: list[dict] = [f.model_dump() for f in (response.parsed or [])]
    high_risks = [f for f in flags if f.get("severity") == "high"]
    needs_review = len(high_risks) >= 2

    sources = format_sources(state["retrieved_chunks"])
    print(f"[Risk] Found {len(flags)} flags, needs_human_review={needs_review}")
    return {
        **state,
        "risk_flags": flags,
        "needs_human_review": needs_review,
        "sources": sources
    }


# ── Node 3c: Memo Drafter ──────────────────────────────────────────────────────

def memo_drafter(state: AgentState) -> AgentState:
    """Draft a structured approval memo from retrieved context."""
    context = format_context(state["retrieved_chunks"])
    prompt = f"""You are a procurement analyst. Draft a concise approval memo based on the contract excerpts below.

Structure the memo exactly as follows:

PROCUREMENT APPROVAL MEMO
─────────────────────────
Vendor: [vendor name]
Contract Type: [type of contract]
Contract Value: [value and currency]
Term: [duration]

KEY TERMS:
- Payment: [payment terms summary]
- SLA / Performance: [service level commitments]
- IP Ownership: [who owns IP]
- Termination: [notice period and conditions]
- Liability Cap: [liability limit]

RISK FLAGS:
[List any concerning clauses with severity — High / Medium / Low]

RECOMMENDATION:
[1-2 sentences: approve / approve with conditions / escalate to legal]

Sources reviewed: [list the contract files referenced]

Contract excerpts:
{context}"""

    response = _llm_call(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    sources = format_sources(state["retrieved_chunks"])
    print(f"[Memo] Draft generated")
    return {**state, "memo": response.text.strip(), "sources": sources}


# ── Router ─────────────────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    """Route to the correct processing node based on intent."""
    return state["intent"]


# ── Graph Assembly ─────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("rag_retriever", rag_retriever)
    graph.add_node("qa_synthesiser", qa_synthesiser)
    graph.add_node("risk_scanner", risk_scanner)
    graph.add_node("memo_drafter", memo_drafter)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "rag_retriever")
    graph.add_conditional_edges(
        "rag_retriever",
        route_by_intent,
        {
            "qa": "qa_synthesiser",
            "risk_scan": "risk_scanner",
            "draft_memo": "memo_drafter"
        }
    )
    graph.add_edge("qa_synthesiser", END)
    graph.add_edge("risk_scanner", END)
    graph.add_edge("memo_drafter", END)

    return graph.compile()


# ── Public Interface ───────────────────────────────────────────────────────────

agent = build_graph()

def run_agent(query: str, source_filter: str = "", override_intent: str = "") -> AgentState:
    """Run the agent with a user query. Returns the full state."""
    initial_state: AgentState = {
        "query": query,
        "intent": "",
        "override_intent": override_intent,
        "source_filter": source_filter,
        "retrieved_chunks": [],
        "answer": "",
        "risk_flags": [],
        "memo": "",
        "sources": [],
        "needs_human_review": False,
    }
    result = agent.invoke(initial_state)
    return result


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1 — Q&A")
    print("=" * 60)
    r = run_agent("What is the auto-renewal notice period in the CloudSync contract?")
    print(r["answer"])
    print("\nSources:", r["sources"])

    print("\n" + "=" * 60)
    print("TEST 2 — Risk Scan")
    print("=" * 60)
    r = run_agent("Scan this contract for risks and red flags")
    for flag in r["risk_flags"]:
        print(f"  [{flag.get('severity','?').upper()}] {flag.get('risk_type')} — {flag.get('clause_summary')}")
    print("Needs human review:", r["needs_human_review"])

    print("\n" + "=" * 60)
    print("TEST 3 — Memo Draft")
    print("=" * 60)
    r = run_agent("Draft an approval memo for the CloudSync contract")
    print(r["memo"])
