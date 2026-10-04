"""Split a document's text into passages for retrieval.

`documents.raw_text` is a list of paragraph blocks joined by blank lines (see
scrapers.rbi.html_to_blocks). Chunks are runs of whole blocks, so a chunk never cuts a
paragraph unless the paragraph alone is longer than the limit. Each chunk keeps:

- exact character offsets into raw_text (raw_text[char_start:char_end] == text), used to
  highlight the cited passage;
- the heading it sits under ("Chapter III > 11. Cooling-off period"), used as context when
  embedding and shown in citations.
"""

import re
from dataclasses import dataclass

BLOCK_SEP = "\n\n"
MAX_WORDS = 220  # bge-small reads up to 512 tokens; leave room for the heading prefix
MIN_WORDS = 40  # merge tiny trailing chunks into the previous one

# Headings RBI uses: "Chapter III – Outsourcing", "CHAPTER I", "Section C", "Annex II",
# "11. Cooling-off period", "A. Board-Approved Policy".
_CHAPTER_RE = re.compile(r"^(chapter|part|annex(ure)?|appendix|schedule)\b", re.I)
_SECTION_RE = re.compile(r"^(\d{1,3}[A-Z]?\.|[A-Z]\.)\s+\S")


@dataclass
class ChunkSpan:
    ord: int
    text: str
    char_start: int
    char_end: int
    heading: str | None


def _blocks_with_offsets(text: str) -> list[tuple[str, int, int]]:
    out, pos = [], 0
    for block in text.split(BLOCK_SEP):
        out.append((block, pos, pos + len(block)))
        pos += len(block) + len(BLOCK_SEP)
    return out


def _heading_kind(block: str) -> str | None:
    """'chapter' or 'section' for short heading-like blocks, else None."""
    first = block.split("\n")[0]
    if len(first.split()) > 14 or first.endswith((".", ";", ":")) and len(first.split()) > 3:
        return None  # sentences, not headings
    if _CHAPTER_RE.match(first):
        return "chapter"
    if _SECTION_RE.match(first):
        return "section"
    return None


def _words(s: str) -> int:
    return len(s.split())


def _split_long(block: str, start: int, max_words: int) -> list[tuple[str, int, int]]:
    """Split one over-long block at sentence ends (or line breaks), keeping offsets."""
    pieces, piece_start, count = [], 0, 0
    for m in re.finditer(r"[^.;\n]+(?:[.;\n]+|$)", block):
        count += _words(m.group(0))
        if count >= max_words:
            pieces.append((piece_start, m.end()))
            piece_start, count = m.end(), 0
    if piece_start < len(block):
        pieces.append((piece_start, len(block)))
    out = []
    for a, b in pieces:
        seg = block[a:b]
        lead = len(seg) - len(seg.lstrip())
        seg = seg.strip()
        if seg:
            out.append((seg, start + a + lead, start + a + lead + len(seg)))
    return out


def chunk_text(
    text: str, max_words: int = MAX_WORDS, min_words: int = MIN_WORDS
) -> list[ChunkSpan]:
    chapter: str | None = None
    section: str | None = None
    groups: list[tuple[int, int, str | None]] = []  # (start, end, heading)
    cur_start = cur_end = None
    cur_words = 0
    cur_heading: str | None = None

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_words
        if cur_start is not None:
            groups.append((cur_start, cur_end, cur_heading))
        cur_start = cur_end = None
        cur_words = 0

    for block, start, end in _blocks_with_offsets(text):
        if not block.strip():
            continue
        kind = _heading_kind(block)
        if kind == "chapter":
            flush()
            chapter, section = block.split("\n")[0], None
        elif kind == "section":
            flush()
            section = block.split("\n")[0]
        heading = " > ".join(h for h in (chapter, section) if h) or None

        n = _words(block)
        if n > max_words:
            flush()
            for _, s, e in _split_long(block, start, max_words):
                groups.append((s, e, heading))
            continue
        if cur_start is not None and cur_words + n > max_words:
            flush()
        if cur_start is None:
            cur_start, cur_heading = start, heading
        cur_end = end
        cur_words += n
    flush()

    limit = int(max_words * 1.25)
    # A tiny chunk (often just a heading, or a run of table-of-contents lines) is merged
    # forward into the chunk after it, which then starts with that heading...
    forward: list[tuple[int, int, str | None]] = []
    carry: int | None = None
    for i, (s, e, h) in enumerate(groups):
        if carry is not None:
            s, carry = carry, None
        if _words(text[s:e]) < min_words and i < len(groups) - 1:
            carry = s
            continue
        forward.append((s, e, h))
    if carry is not None:  # tiny chunk at the very end
        forward.append((carry, groups[-1][1], groups[-1][2]))

    # ...and anything still tiny (or grown past the limit) is settled against its neighbour.
    merged: list[tuple[int, int, str | None]] = []
    for s, e, h in forward:
        if merged and _words(text[s:e]) < min_words and _words(text[merged[-1][0] : e]) <= limit:
            merged[-1] = (merged[-1][0], e, merged[-1][2])
            continue
        merged.append((s, e, h))
    return [ChunkSpan(i, text[s:e], s, e, h) for i, (s, e, h) in enumerate(merged)]
