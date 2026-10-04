"""The question-answering agent as a LangGraph workflow.

    understand -> retrieve -> grade --answerable--> generate -> verify -> END
                     ^          |
                     |       not answerable, tries left
                     +-- rewrite <-+
                                |
                                +-- no tries left --> abstain -> END

- understand: the date asked about (explicit, read from the question, or today).
- retrieve: hybrid passages in force on that date, plus amendments of what was found.
- grade: the model keeps only passages that bear on the question and says whether they
  answer it; if not, it proposes a better search query (at most MAX_REWRITES retries).
  Out-of-scope questions end here, as an abstention.
- generate / verify: quoted sentences, each quote checked against its passage.
"""

import json
import logging
from datetime import date
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from niyam.rag.answer import Answer, to_passages
from niyam.rag.dates import resolve_as_of
from niyam.rag.llm import LLM
from niyam.rag.prompt import build_messages, parse_reply
from niyam.rag.verify import PassageRef, verify_sentences
from niyam.retrieval.embeddings import Embedder
from niyam.retrieval.hybrid import search_chunks
from niyam.retrieval.keyword import SearchFilters
from niyam.retrieval.temporal import expand_with_amendments

log = logging.getLogger(__name__)

MAX_REWRITES = 2
PASSAGES = 8
GRADE_CHARS = 1500  # show the grader (nearly) whole passages, not just their opening

GRADER = """You check whether retrieved passages can answer a question about Indian \
financial regulation (RBI). For each passage label, decide if it is relevant to the \
question. Reply with JSON only:
{"relevant": ["P1", ...], "answerable": true|false, "better_query": "..."}
"answerable" is true only if the relevant passages contain the answer. When it is false
and the question is about RBI regulation, give a better search query (key terms, the
regulated entity, the topic); otherwise "better_query" is ""."""


class AgentState(TypedDict, total=False):
    question: str
    as_of: date
    as_of_source: str  # request | question | today
    query: str
    rewrites: int
    passages: list[PassageRef]
    answerable: bool
    better_query: str
    trace: list[str]
    answer: Answer


def build_agent(session: Session, llm: LLM, embedder: Embedder, grader: LLM | None = None):
    grader = grader or llm

    def understand(state: AgentState) -> AgentState:
        as_of, source = resolve_as_of(state["question"], state.get("as_of"))
        return {
            "as_of": as_of,
            "as_of_source": source,
            "query": state["question"],
            "rewrites": 0,
            "trace": [f"as of {as_of} ({source})"],
        }

    def retrieve(state: AgentState) -> AgentState:
        hits = search_chunks(
            session,
            state["query"],
            embedder,
            k=PASSAGES,
            filters=SearchFilters(as_of=state["as_of"]),
        )
        hits = expand_with_amendments(session, state["query"], hits, state["as_of"])
        passages = to_passages(session, hits, state["as_of"])
        return {
            "passages": passages,
            "trace": state["trace"] + [f"retrieved {len(passages)} for {state['query']!r}"],
        }

    def grade(state: AgentState) -> AgentState:
        if not state["passages"]:
            return {
                "answerable": False,
                "better_query": "",
                "trace": state["trace"] + ["no passages"],
            }
        listing = "\n\n".join(
            f"[{p.label}] {p.title} | {p.heading or ''}\n{p.text[:GRADE_CHARS]}"
            for p in state["passages"]
        )
        reply = grader.complete(
            [
                {"role": "system", "content": GRADER},
                {"role": "user", "content": f"Question: {state['question']}\n\n{listing}"},
            ]
        )
        try:
            data = json.loads(reply[reply.find("{") : reply.rfind("}") + 1])
        except ValueError:
            log.warning("unparseable grader reply; keeping all passages")
            data = {"relevant": [p.label for p in state["passages"]], "answerable": True}
        keep = set(data.get("relevant") or [])
        passages = [p for p in state["passages"] if p.label in keep] or state["passages"]
        answerable = bool(data.get("answerable")) and bool(keep)
        return {
            "passages": passages,
            "answerable": answerable,
            "better_query": str(data.get("better_query") or "").strip(),
            "trace": state["trace"] + [f"graded: {len(keep)} relevant, answerable={answerable}"],
        }

    def after_grade(state: AgentState) -> str:
        if state["answerable"]:
            return "generate"
        if state["rewrites"] < MAX_REWRITES and state.get("better_query"):
            return "rewrite"
        return "abstain"

    def rewrite(state: AgentState) -> AgentState:
        return {
            "query": state["better_query"],
            "rewrites": state["rewrites"] + 1,
            "trace": state["trace"] + [f"rewrite -> {state['better_query']!r}"],
        }

    def generate_and_verify(state: AgentState) -> AgentState:
        # Relabel the kept passages P1..Pn so the prompt has no gaps.
        passages = [
            PassageRef(**{**vars(p), "label": f"P{i}"}) for i, p in enumerate(state["passages"], 1)
        ]
        reply = llm.complete(
            build_messages(state["question"], passages, state["as_of"].isoformat())
        )
        base = Answer(state["question"], state["as_of"], True, [], passages)
        try:
            data = parse_reply(reply)
        except ValueError:
            base.note = "The model reply was not valid."
            return {"answer": base, "trace": state["trace"] + ["invalid reply"]}
        if data.get("abstained"):
            base.note = " ".join(str(s.get("text", "")) for s in data["sentences"]) or None
            return {"answer": base, "trace": state["trace"] + ["model abstained"]}
        sentences = verify_sentences(data["sentences"], passages)
        supported = [s for s in sentences if s.supported]
        dropped = [s.text for s in sentences if not s.supported]
        if not supported:
            base.note = "None of the answer could be matched to the source text."
            base.unsupported = dropped
            return {"answer": base, "trace": state["trace"] + ["no verified sentences"]}
        answer = Answer(
            state["question"], state["as_of"], False, supported, passages, unsupported=dropped
        )
        return {
            "answer": answer,
            "trace": state["trace"] + [f"{len(supported)} verified, {len(dropped)} dropped"],
        }

    def abstain(state: AgentState) -> AgentState:
        note = (
            "The sources found don't answer this question."
            if state.get("passages")
            else "No matching RBI documents were found."
        )
        answer = Answer(
            state["question"], state["as_of"], True, [], state.get("passages", []), note
        )
        return {"answer": answer, "trace": state["trace"] + ["abstain"]}

    g = StateGraph(AgentState)
    g.add_node("understand", understand)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate_and_verify)
    g.add_node("abstain", abstain)
    g.add_edge(START, "understand")
    g.add_edge("understand", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["generate", "rewrite", "abstain"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("abstain", END)
    return g.compile()


def run_agent(
    session: Session,
    question: str,
    llm: LLM,
    embedder: Embedder,
    as_of: date | None = None,
    grader: LLM | None = None,
) -> AgentState:
    state: AgentState = {"question": question}
    if as_of:
        state["as_of"] = as_of
    return build_agent(session, llm, embedder, grader).invoke(state)
