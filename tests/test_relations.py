from niyam.ingest.relations import (
    FoundRelation,
    RefIndex,
    link_relations,
    merge,
    normalize_ref,
    subject,
    text_relations,
)

REFS = {
    "100": "DOR.AML.REC.44/14.01.001/2023-24",
    "200": "DOR.RRC.REC.302/33-01-010/2025-26",
    "300": "DOR.STR.REC.13/13.03.00/2024-25",
}
TITLES = {
    "100": "Amendment to the KYC Master Direction",
    "200": "Consolidation of Regulations – Withdrawal of circulars",
    "300": "Key Facts Statement (KFS) for Loans & Advances",
    "500": "Reserve Bank of India (Commercial Banks – Credit Risk Management) Directions, 2025",
    "501": "Reserve Bank of India (Commercial Banks - Credit Risk Management) - Second Amendment "
    "Directions, 2026",
    "502": "Reserve Bank of India (Commercial Banks – Capital Charge) Directions, 2025",
}


def index() -> RefIndex:
    return RefIndex(REFS)


def test_normalize_ref_ignores_spacing_and_no():
    assert normalize_ref("DOR.AML.REC.No. 44 / 14.01.001 / 2023-24") == normalize_ref(
        "DOR.AML.REC.44/14.01.001/2023-24"
    )


def test_index_drops_short_and_ambiguous_numbers():
    idx = RefIndex({"1": "A/1", "2": "DOR.X.1/01.01.001/2024-25", "3": "DOR.X.1/01.01.001/2024-25"})
    assert idx.unique == {}


def test_supersession_and_withdrawal_paragraphs():
    text = (
        "1. In supersession of circular DOR.STR.REC.13/13.03.00/2024-25, banks shall ...\n\n"
        "2. Circular DOR.AML.REC.44/14.01.001/2023-24 shall stand withdrawn."
    )
    rels = {(r.dst, r.type) for r in text_relations("900", "Some circular", text, index(), TITLES)}
    assert rels == {("300", "supersedes"), ("100", "repeals")}


def test_cited_as_instrument_is_only_a_reference():
    text = (
        "17. The existing instructions stand repealed as communicated circular "
        "DOR.RRC.REC.302/33-01-010/2025-26 dated November 28, 2025."
    )
    (r,) = text_relations("900", "Some Directions, 2025", text, index(), TITLES)
    assert (r.dst, r.type) == ("200", "refers")


def test_footnote_means_the_cited_circular_amended_this_document():
    text = "12 Amended vide circular DOR.AML.REC.44/14.01.001/2023-24 dated October 17, 2023."
    (r,) = text_relations("11566", "KYC Master Direction", text, index(), TITLES)
    assert (r.src, r.dst, r.type) == ("100", "11566", "amends")


def test_plain_mention_is_a_reference():
    text = "Banks shall comply with circular DOR.STR.REC.13/13.03.00/2024-25 on KFS."
    (r,) = text_relations("900", "Master Circular – Housing Finance", text, index(), TITLES)
    assert r.type == "refers"


def test_subject_of_titles():
    assert (
        subject(TITLES["500"])
        == subject(TITLES["501"])
        == ("commercial banks-credit risk management")
    )
    assert subject("Some circular") is None


def test_amendment_link_amends_only_its_own_subject():
    rels = link_relations("501", TITLES["501"], [500, 502, 999], TITLES)
    assert {(r.dst, r.type) for r in rels} == {("500", "amends"), ("502", "refers")}


def test_non_amendment_links_are_references():
    (r,) = link_relations("500", TITLES["500"], [502], TITLES)
    assert r.type == "refers"


def test_merge_keeps_strongest_and_drops_redundant_refers():
    rels = [
        FoundRelation("1", "2", "refers", "link", 0.5, "a"),
        FoundRelation("1", "2", "amends", "link", 0.9, "b"),
        FoundRelation("1", "2", "amends", "regex", 0.7, "c"),
        FoundRelation("1", "3", "refers", "regex", 0.5, "d"),
    ]
    merged = merge(rels)
    assert {(r.dst, r.type, r.confidence) for r in merged} == {
        ("2", "amends", 0.9),
        ("3", "refers", 0.5),
    }
