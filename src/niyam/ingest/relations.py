"""Find relations between documents: which circular amends, supersedes or repeals which.

Sources, all from text and pages already stored (no network):

- Reference numbers. RBI documents cite each other by reference number
  ("DOR.AML.REC.44/14.01.001/2023-24"). Every stored document's number is indexed; each
  paragraph is scanned for known numbers, and the verbs in that paragraph decide the type
  ("in supersession of" -> supersedes, "stand withdrawn / repealed" -> repeals,
  "amended vide" / "inserted vide" / an Amendment title -> amends, otherwise refers).
- Hyperlinks. Pages link to the documents they amend (NotificationUser.aspx?Id=N);
  an "Amendment" document linking to a Master Direction amends it.

Relations point from the newer document (src) to the one it acts on (dst), except
footnotes in a Master Direction ("Amended vide circular X dated ..."), where the cited
circular X is the src and the Master Direction the dst.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

YEAR_RE = re.compile(r"\d{4}-\d{2}\b")

SUPERSEDES_RE = re.compile(r"in supersession of|supersedes|superseded", re.I)
REPEALS_RE = re.compile(
    r"stands? (?:withdrawn|repealed)|are (?:hereby )?(?:withdrawn|repealed)", re.I
)
FOOTNOTE_RE = re.compile(
    r"^\s*\d{0,3}\s*(?:amended|inserted|substituted|deleted|added) vide\b", re.I
)
AMENDS_RE = re.compile(r"\bamend(?:s|ed|ment)\b", re.I)

# Words just before a cited number that mark it as the instrument ("as communicated vide
# circular X"), matched against the space-free normalised text.
INSTRUMENT_RE = re.compile(r"vide|communicated|conveyed")
MIN_REF_CHARS = 14  # shorter normalised numbers collide too easily
SUBJECT_RE = re.compile(r"\(([^()]{8,})\)")


def subject(title: str) -> str | None:
    """The parenthetical subject of an RBI Direction title, normalised:
    'Reserve Bank of India (Commercial Banks – Credit Risk Management) - Second Amendment
    Directions, 2026' -> 'commercial banks-credit risk management'."""
    m = SUBJECT_RE.search(title)
    if not m:
        return None
    s = re.sub(r"\s+", " ", m.group(1).lower().replace("–", "-").replace("—", "-"))
    return re.sub(r"\s*-\s*", "-", s).strip()


def normalize_ref(ref: str) -> str:
    """Lower-case, drop 'No.' and every space, so formatting variants compare equal."""
    s = ref.lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"\bnos?\b\.?", "", s)
    return re.sub(r"[\s,]+", "", s).strip(".")


@dataclass(frozen=True)
class FoundRelation:
    src: str  # source_id of the acting document
    dst: str  # source_id of the document acted on
    type: str  # amends | supersedes | repeals | refers
    method: str  # regex | link
    confidence: float
    evidence: str


class RefIndex:
    """Known reference numbers, grouped by financial-year suffix for fast lookup."""

    def __init__(self, refs: dict[str, str]):  # source_id -> circular_no
        by_norm: dict[str, set[str]] = defaultdict(set)
        for sid, ref in refs.items():
            n = normalize_ref(ref or "")
            if len(n) >= MIN_REF_CHARS and "/" in n:
                by_norm[n].add(sid)
        # A number shared by two documents can't be resolved reliably.
        self.unique = {n: next(iter(s)) for n, s in by_norm.items() if len(s) == 1}
        self.by_year: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for n, sid in self.unique.items():
            m = YEAR_RE.search(n)
            self.by_year[m.group(0) if m else ""].append((n, sid))

    def find(self, paragraph: str) -> dict[str, bool]:
        """Documents cited in the paragraph -> True when cited as the instrument
        ("as communicated vide circular X") rather than as the sentence's subject."""
        years = set(YEAR_RE.findall(paragraph))
        if not years:
            return {}
        norm = normalize_ref(paragraph)
        found: dict[str, bool] = {}
        for y in years:
            for n, sid in self.by_year.get(y, ()):
                pos = norm.find(n)
                if pos >= 0:
                    found[sid] = bool(INSTRUMENT_RE.search(norm[max(0, pos - 40) : pos]))
        return found


def classify(paragraph: str, src_title: str) -> tuple[str, float]:
    if SUPERSEDES_RE.search(paragraph):
        return "supersedes", 0.9
    if REPEALS_RE.search(paragraph):
        return "repeals", 0.9
    if AMENDS_RE.search(paragraph) or "amendment" in src_title.lower():
        return "amends", 0.7
    return "refers", 0.5


def text_relations(
    source_id: str,
    title: str,
    text: str,
    index: RefIndex,
    titles: dict[str, str] | None = None,
) -> list[FoundRelation]:
    out = []
    for para in text.split("\n\n"):
        cited = index.find(para)
        cited.pop(source_id, None)
        if not cited:
            continue
        evidence = " ".join(para.split())[:500]
        if FOOTNOTE_RE.search(para):
            # "Amended vide circular X dated ..." inside this document: X amended this one.
            for t in cited:
                out.append(FoundRelation(t, source_id, "amends", "regex", 0.9, evidence))
            continue
        rtype, conf = classify(para, title)
        own = subject(title)
        for t, instrument in cited.items():
            kind = rtype
            if instrument:  # the circular through which something was done, not its target
                kind = "refers"
            elif kind == "amends" and own and titles and subject(titles.get(t, "")) != own:
                kind = "refers"  # an amendment amends its own subject; others are citations
            out.append(
                FoundRelation(source_id, t, kind, "regex", conf if kind == rtype else 0.5, evidence)
            )
    return out


def link_relations(
    source_id: str, title: str, linked_ids: list[int], titles: dict[str, str]
) -> list[FoundRelation]:
    """Hyperlinks from this page to other stored documents. An Amendment amends the linked
    document with the same subject ("(Commercial Banks – Credit Risk Management)"); other
    links are references."""
    out = []
    own = subject(title) if "amendment" in title.lower() else None
    for lid in linked_ids:
        dst = str(lid)
        if dst == source_id or dst not in titles:
            continue
        if own and subject(titles[dst]) == own:
            out.append(FoundRelation(source_id, dst, "amends", "link", 0.9, f"link in {title}"))
        else:
            out.append(FoundRelation(source_id, dst, "refers", "link", 0.5, f"link in {title}"))
    return out


TYPE_RANK = {"supersedes": 3, "repeals": 3, "amends": 2, "refers": 1}


def merge(relations: list[FoundRelation]) -> list[FoundRelation]:
    """One relation per (src, dst, type); keep the most confident. Drop 'refers' when a
    stronger relation exists for the same pair."""
    best: dict[tuple[str, str, str], FoundRelation] = {}
    for r in relations:
        key = (r.src, r.dst, r.type)
        if key not in best or r.confidence > best[key].confidence:
            best[key] = r
    strong = {(r.src, r.dst) for r in best.values() if r.type != "refers"}
    return [r for r in best.values() if r.type != "refers" or (r.src, r.dst) not in strong]


AUTO_METHODS = ("regex", "link")


def extract_all(session, data_dir) -> list[FoundRelation]:
    """Relations for every stored document, from its text and its cached page's links."""
    from pathlib import Path

    from sqlalchemy import select

    from niyam.db.models import Document
    from niyam.ingest.scrapers.rbi import parse_detail

    docs = session.execute(
        select(
            Document.source_id,
            Document.title,
            Document.raw_text,
            Document.circular_no,
            Document.source_path,
        ).where(Document.source_id.is_not(None))
    ).all()
    index = RefIndex({d.source_id: d.circular_no for d in docs})
    titles = {d.source_id: d.title for d in docs}
    found: list[FoundRelation] = []
    for d in docs:
        found += text_relations(d.source_id, d.title, d.raw_text or "", index, titles)
        if d.source_path:
            path = Path(data_dir) / d.source_path
            if path.exists():
                try:
                    links = parse_detail(path.read_text(encoding="utf-8"), int(d.source_id))
                except ValueError:
                    continue
                found += link_relations(d.source_id, d.title, links.linked_ids, titles)
    return merge(found)


def save_relations(session, relations: list[FoundRelation]) -> int:
    """Replace all automatically extracted relations with `relations` in one transaction."""
    from sqlalchemy import delete, select

    from niyam.db.models import Document, Relation

    ids = dict(session.execute(select(Document.source_id, Document.id)).all())
    session.execute(delete(Relation).where(Relation.method.in_(AUTO_METHODS)))
    rows = [
        Relation(
            src_doc_id=ids[r.src],
            dst_doc_id=ids[r.dst],
            type=r.type,
            method=r.method,
            confidence=r.confidence,
            evidence_text=r.evidence,
        )
        for r in relations
        if r.src in ids and r.dst in ids
    ]
    session.add_all(rows)
    session.commit()
    return len(rows)
