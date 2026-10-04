from niyam.ingest.chunk import chunk_text


def para(n: int, word: str = "word") -> str:
    return " ".join([word] * n)


def doc(*blocks: str) -> str:
    return "\n\n".join(blocks)


def test_offsets_point_back_into_the_text():
    text = doc("Chapter I – Preliminary", "1. Short title", para(50), para(80), para(120))
    for c in chunk_text(text, max_words=100, min_words=10):
        assert text[c.char_start : c.char_end] == c.text


def test_blocks_are_grouped_up_to_the_limit_and_never_split():
    text = doc(para(60, "a"), para(60, "b"), para(60, "c"))
    chunks = chunk_text(text, max_words=130, min_words=10)
    assert [c.text.split()[0] for c in chunks] == ["a", "c"]
    assert chunks[0].text == doc(para(60, "a"), para(60, "b"))


def test_headings_start_new_chunks_and_are_tracked():
    text = doc(
        "Chapter III – Digital Lending",
        "11. Cooling-off period",
        para(50, "exit"),
        "12. Grievance redressal",
        para(50, "complaint"),
        "Chapter IV – Gold Loans",
        para(50, "gold"),
    )
    chunks = chunk_text(text, max_words=200, min_words=10)
    assert [c.heading for c in chunks] == [
        "Chapter III – Digital Lending > 11. Cooling-off period",
        "Chapter III – Digital Lending > 12. Grievance redressal",
        "Chapter IV – Gold Loans",
    ]
    # The heading text itself is the start of the chunk that follows it.
    assert chunks[0].text.startswith("Chapter III – Digital Lending\n\n11. Cooling-off period")


def test_sentences_are_not_mistaken_for_headings():
    text = doc(
        "2. Accordingly, in exercise of the powers conferred by section 35A of the Act, "
        "the Reserve Bank hereby issues these directions.",
        para(50),
    )
    assert chunk_text(text, max_words=200, min_words=10)[0].heading is None


def test_long_block_is_split_at_sentence_ends():
    sentences = " ".join(f"Sentence number {i} has some words in it." for i in range(60))
    chunks = chunk_text(sentences, max_words=100, min_words=10)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 110 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)
    for c in chunks:
        assert sentences[c.char_start : c.char_end] == c.text


def test_tiny_chunks_are_merged():
    text = doc(
        "Table of Contents",
        "Chapter I",
        "Chapter II",
        "Chapter I – Preliminary",
        para(80, "body"),
    )
    chunks = chunk_text(text, max_words=200, min_words=40)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Table of Contents")


def test_empty_text():
    assert chunk_text("") == []
