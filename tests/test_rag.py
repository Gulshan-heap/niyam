import json
from datetime import date

import pytest

from niyam.rag.prompt import build_messages, parse_reply
from niyam.rag.verify import PassageRef, locate, verify_sentences

PASSAGE_TEXT = (
    "11. Cooling-off period\n\n(1) The borrower shall be given an explicit option to exit a "
    "digital loan by paying the principal and the proportionate APR without any penalty during "
    "an initial “cooling-off period”. The cooling off period shall be determined by the bank in "
    "terms of its credit policy, subject to the period so determined not being less than one day."
)


def passage(label="P1", text=PASSAGE_TEXT, char_start=5000) -> PassageRef:
    return PassageRef(
        label=label,
        chunk_id=42,
        source_id="13156",
        title="Reserve Bank of India (Commercial Banks – Credit Facilities) Directions, 2025",
        heading="Chapter III > 11. Cooling-off period",
        text=text,
        char_start=char_start,
        url="https://example.test/13156",
        issued_date="2025-11-28",
        in_force=True,
    )


# ---------- locating quotes ----------


def test_exact_quote_is_located():
    quote = "not being less than one day"
    score, start, end = locate(quote, PASSAGE_TEXT)
    assert score == 100.0
    assert PASSAGE_TEXT[start:end] == quote


def test_lightly_paraphrased_quote_still_matches():
    # curly vs straight quotes and doubled spaces should not break a verbatim quote
    quote = (
        "exit a digital loan by paying the principal and the proportionate APR  without any "
        'penalty during an initial "cooling-off period"'
    )
    score, start, end = locate(quote, PASSAGE_TEXT)
    assert score >= 90
    assert "exit a digital loan" in PASSAGE_TEXT[start:end]


def test_invented_quote_does_not_match():
    score, _, _ = locate("borrowers may cancel within thirty days for a full refund", PASSAGE_TEXT)
    assert score < 90


def test_too_short_quote_is_rejected():
    assert locate("one day", PASSAGE_TEXT) == (0.0, None, None)


# ---------- verifying sentences ----------


def test_verified_citation_gets_absolute_document_offsets():
    raw = [
        {
            "text": "Banks must allow a cooling-off period of at least one day.",
            "citations": [{"passage": "P1", "quote": "not being less than one day"}],
        }
    ]
    (s,) = verify_sentences(raw, [passage(char_start=5000)])
    assert s.supported
    c = s.citations[0]
    assert c.source_id == "13156" and c.chunk_id == 42
    local = PASSAGE_TEXT.find("not being less than one day")
    assert (c.doc_char_start, c.doc_char_end) == (5000 + local, 5000 + local + 27)


def test_fabricated_quote_and_unknown_passage_are_unsupported():
    raw = [
        {
            "text": "A.",
            "citations": [{"passage": "P1", "quote": "a thirty day free-look period applies"}],
        },
        {"text": "B.", "citations": [{"passage": "P9", "quote": "not being less than one day"}]},
        {"text": "C.", "citations": []},
    ]
    sentences = verify_sentences(raw, [passage()])
    assert [s.supported for s in sentences] == [False, False, False]


def test_passage_label_brackets_are_tolerated():
    raw = [
        {"text": "X.", "citations": [{"passage": "[P1]", "quote": "not being less than one day"}]}
    ]
    assert verify_sentences(raw, [passage()])[0].supported


# ---------- prompt ----------


def test_prompt_numbers_passages_and_states_the_date():
    msgs = build_messages("Can I exit a digital loan?", [passage(), passage("P2")], "2024-06-01")
    assert msgs[0]["role"] == "system" and "VERBATIM" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "[P1] Reserve Bank of India" in user and "[P2]" in user
    assert "issued 2025-11-28" in user and "in force" in user
    assert "in force on 2024-06-01" in user


def test_parse_reply_tolerates_fences():
    payload = {"abstained": False, "sentences": [{"text": "x", "citations": []}]}
    assert parse_reply(f"```json\n{json.dumps(payload)}\n```") == payload
    assert parse_reply('{"sentences": []}') == {"sentences": [], "abstained": False}
    with pytest.raises(ValueError):
        parse_reply("I cannot answer that.")


def test_date_type_is_iso_in_prompt():
    msgs = build_messages("q", [passage()], date(2024, 6, 1).isoformat())
    assert "2024-06-01" in msgs[1]["content"]


def test_models_without_a_provider_key_are_skipped(monkeypatch):
    from niyam.rag import llm

    monkeypatch.setattr(llm, "load_dotenv", lambda **kw: None)
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert llm.has_key("groq/openai/gpt-oss-120b")
    assert not llm.has_key("gemini/gemini-2.5-flash")
    assert llm.has_key("ollama/llama3")  # local providers need no key
    chosen = llm.LiteLLM("gemini/gemini-2.5-flash", ["groq/qwen/qwen3.8-27b"])
    assert chosen.models == ["groq/qwen/qwen3.8-27b"]


def test_tracing_is_off_without_langfuse_keys(monkeypatch):
    from niyam.rag import llm

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert llm.enable_tracing() is False
