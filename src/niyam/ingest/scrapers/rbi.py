"""RBI website scraper: notification listings, the Master Directions index, and detail pages.

PDFs live on rbidocs.rbi.org.in behind a bot challenge, so we don't download them. The
detail page at NotificationUser.aspx?Id=N carries the full text, so it is the source of truth.
Master Directions share the same id space (BS_ViewMasDirections.aspx?id=N is the same
document), so every document is stored under its NotificationUser URL.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from niyam.ingest.http import PoliteClient

BASE_URL = "https://www.rbi.org.in/Scripts/"
LISTING_URL = BASE_URL + "NotificationUser.aspx"
MD_INDEX_URL = BASE_URL + "BS_ViewMasterDirections.aspx"

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
DATE_RE = re.compile(rf"\b({_MONTHS})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b")
# Only the "(Updated as on <date>)" stamps RBI puts in titles, not "updated as on" in prose.
UPDATED_RE = re.compile(rf"\(\s*Updated as on\s+({_MONTHS})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})")
RBI_NO_RE = re.compile(r"\bRBI/(?:[A-Za-z]+/)?\d{4}-\d{2,4}/\d+")
WITHDRAWN_RE = re.compile(r"Withdrawn(\d{2})(\d{2})(\d{4})\.jpg", re.I)
DOC_LINK_RE = re.compile(r"(?:NotificationUser|BS_ViewMasDirections)\.aspx\?id=(\d+)", re.I)

# Reference-number prefixes -> issuing department. Old names are kept for older circulars.
DEPARTMENTS = {
    "DOR": "Department of Regulation",
    "DOS": "Department of Supervision",
    "DPSS": "Department of Payment and Settlement Systems",
    "FED": "Foreign Exchange Department",
    "DCM": "Department of Currency Management",
    "FIDD": "Financial Inclusion and Development Department",
    "DGBA": "Department of Government and Bank Accounts",
    "CEPD": "Consumer Education and Protection Department",
    "FMRD": "Financial Markets Regulation Department",
    "IDMD": "Internal Debt Management Department",
    "DSIM": "Department of Statistics and Information Management",
    "DOC": "Department of Communication",
    "DBR": "Department of Banking Regulation",
    "DBOD": "Department of Banking Operations and Development",
    "DNBR": "Department of Non-Banking Regulation",
    "DNBS": "Department of Non-Banking Supervision",
    "DCBR": "Department of Co-operative Bank Regulation",
    "UBD": "Urban Banks Department",
    "RPCD": "Rural Planning and Credit Department",
    "DBS": "Department of Banking Supervision",
    "DEIO": "Department of External Investments and Operations",
}
_DEPT_RE = re.compile(r"\b(" + "|".join(DEPARTMENTS) + r")\b", re.I)
_AP_DIR_RE = re.compile(r"A\.\s*P\.\s*\(DIR Series\)", re.I)

_BLOCK_TAGS = {
    "p", "div", "li", "tr", "table", "tbody", "thead", "tfoot", "ol", "ul", "dl", "dt", "dd",
    "h1", "h2", "h3", "h4", "h5", "h6", "hr", "blockquote", "center", "pre", "section",
}  # fmt: skip


@dataclass
class ListingEntry:
    rbi_id: int
    title: str
    listed_date: date | None
    pdf_url: str | None
    section: str | None = None  # Master Directions index groups entries by sector
    updated_on: date | None = None  # "(Updated as on ...)" in the title, if any


@dataclass
class DetailPage:
    rbi_id: int
    title: str
    rbi_no: str | None
    ref_no: str | None
    department: str | None
    issued_date: date | None
    updated_on: date | None
    withdrawn_on: date | None
    pdf_url: str | None
    blocks: list[str]
    linked_ids: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.blocks)


def detail_url(rbi_id: int) -> str:
    return f"{BASE_URL}NotificationUser.aspx?Id={rbi_id}&Mode=0"


def parse_date(s: str) -> date | None:
    m = DATE_RE.search(s)
    if not m:
        return None
    month = m.group(1)[:3]
    try:
        return datetime.strptime(f"{month} {m.group(2)} {m.group(3)}", "%b %d %Y").date()
    except ValueError:
        return None


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def _strip_updated(title: str) -> str:
    return _clean(re.sub(r"\(\s*Updated as on[^)]*\)", "", title))


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


# ---------- listings ----------


def parse_form_state(html: str) -> dict[str, str]:
    """Hidden ASP.NET fields needed to post back the year/month filter."""
    soup = _soup(html)
    state = {
        i["name"]: i.get("value", "")
        for i in soup.find_all("input", attrs={"type": "hidden"})
        if i.get("name")
    }
    state["UsrFontCntr$btn"] = ""
    return state


def parse_listing(html: str) -> list[ListingEntry]:
    """Parse a notifications listing or the Master Directions index.

    Both are tables of header rows (a date, or a sector name on the MD index) followed by
    entry rows with a detail link and a PDF link.
    """
    soup = _soup(html)
    entries: list[ListingEntry] = []
    seen: set[int] = set()
    current_date: date | None = None
    section: str | None = None
    for row in soup.select("table.tablebg tr"):
        header = row.find("td", class_="tableheader")
        if header is not None and row.find("a", class_="link2") is None:
            text = _clean(header.get_text(" "))
            if re.fullmatch(rf"({_MONTHS})\.? \d{{1,2}}, \d{{4}}", text):
                current_date = parse_date(text)
            elif text:
                section, current_date = text, None
            continue
        link = row.find("a", class_="link2")
        if link is None:
            continue
        m = DOC_LINK_RE.search(link.get("href", ""))
        if not m or int(m.group(1)) in seen:
            continue
        rbi_id = int(m.group(1))
        seen.add(rbi_id)
        pdf = row.find("a", href=re.compile(r"\.pdf$", re.I))
        raw_title = link.get_text(" ")
        entries.append(
            ListingEntry(
                rbi_id=rbi_id,
                title=_strip_updated(raw_title),
                listed_date=current_date,
                pdf_url=pdf["href"] if pdf else None,
                section=section,
                updated_on=_latest_updated(raw_title),
            )
        )
    return entries


# ---------- detail page ----------


def html_to_blocks(root: Tag) -> list[str]:
    """Flatten messy RBI HTML into paragraphs.

    Paragraph breaks come from block-level tags wherever they are nested (RBI pages wrap
    paragraphs in <span>s); <br> becomes a line break; table cells are joined with " | ".
    """
    blocks: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        lines = []
        for line in "".join(buf).split("\n"):
            line = _clean(line).strip("| ").strip()
            if line:
                lines.append(line)
        if lines:
            blocks.append("\n".join(lines))
        buf.clear()

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                buf.append(str(child))
                continue
            if not isinstance(child, Tag) or child.name in ("script", "style"):
                continue
            if child.name == "br":
                buf.append("\n")
            elif child.name == "sup" and child.find("a", href=re.compile(r"^#")):
                continue  # footnote marker; the footnote itself is kept as a block
            elif child.name in ("td", "th"):
                walk(child)
                buf.append(" | ")
            elif child.name in _BLOCK_TAGS:
                flush()
                walk(child)
                flush()
            else:
                walk(child)

    walk(root)
    flush()
    return blocks


def _department(ref_no: str | None, head: str = "") -> str | None:
    """From the reference prefix (DOR.…, A.P. (DIR Series), FEMA …), else the letterhead."""
    if ref_no:
        if _AP_DIR_RE.search(ref_no) or re.search(r"\bFEMA\b", ref_no):
            return DEPARTMENTS["FED"]
        m = _DEPT_RE.search(ref_no)
        if m:
            return DEPARTMENTS[m.group(1).upper()]
    head = head.lower()
    return next((name for name in DEPARTMENTS.values() if name.lower() in head), None)


def _numbers(blocks: list[str]) -> tuple[str | None, str | None]:
    """RBI number (RBI/2026-27/278) and the department reference on the line after it.

    FEMA notifications have no RBI number, only a "Notification No. FEMA …" line.
    """
    for block in blocks[:8]:
        if block.startswith("Notification No"):
            return None, block.split("\n")[0]
    for block in blocks[:5]:
        m = RBI_NO_RE.search(block)
        if not m:
            continue
        lines = block.split("\n")
        idx = next(i for i, line in enumerate(lines) if m.group(0) in line)
        ref = lines[idx + 1] if idx + 1 < len(lines) else None
        if ref:
            ref = re.sub(r"^Master Direction\s*[-–]?\s*", "", ref).strip() or None
        return m.group(0), ref
    return None, None


def _issued_date(blocks: list[str]) -> date | None:
    """The date line near the top (right-aligned on the page), not a date in the body."""
    for block in blocks[:8]:
        first = block.split("\n")[0]
        if DATE_RE.fullmatch(first.strip(" .")):
            return parse_date(first)
    return None


def _latest_updated(text: str) -> date | None:
    dates = [parse_date(" ".join(m.groups())) for m in UPDATED_RE.finditer(text)]
    dates = [d for d in dates if d]
    return max(dates) if dates else None


def parse_detail(html: str, rbi_id: int) -> DetailPage:
    soup = _soup(html)
    container = soup.find("div", id="NotificationUser")
    if container is None:
        raise ValueError(f"RBI page {rbi_id}: no NotificationUser container")

    title_cell = container.find("td", class_="tableheader", align="center")
    raw_title = _clean(title_cell.get_text(" ")) if title_cell else ""
    title = _strip_updated(raw_title)

    pdf = container.find("a", href=re.compile(r"\.pdf$", re.I))
    content = container.find("tr", class_="tablecontent2")
    if content is None:
        raise ValueError(f"RBI page {rbi_id}: no content row")
    blocks = html_to_blocks(content)

    withdrawn_on = None
    m = WITHDRAWN_RE.search(str(content))
    if m:
        dd, mm, yyyy = (int(g) for g in m.groups())
        withdrawn_on = date(yyyy, mm, dd)

    linked: list[int] = []
    for a in content.find_all("a", href=True):
        lm = DOC_LINK_RE.search(a["href"])
        if lm and int(lm.group(1)) != rbi_id and int(lm.group(1)) not in linked:
            linked.append(int(lm.group(1)))

    rbi_no, ref_no = _numbers(blocks)
    head = "\n".join(blocks[:8])
    return DetailPage(
        rbi_id=rbi_id,
        title=title,
        rbi_no=rbi_no,
        ref_no=ref_no,
        department=_department(ref_no, head),
        issued_date=_issued_date(blocks),
        updated_on=_latest_updated(raw_title + "\n" + head),
        withdrawn_on=withdrawn_on,
        pdf_url=pdf["href"] if pdf else None,
        blocks=blocks,
        linked_ids=linked,
    )


def is_master_direction(title: str) -> bool:
    if re.match(r"Master Directions?\b", title, re.I):
        return True
    # Post-2025 consolidated rulebooks: "Reserve Bank of India (...) Directions, 2025"
    return bool(re.search(r"\bDirections?, \d{4}\b", title)) and "Amendment" not in title


# ---------- scraper ----------


class RbiScraper:
    def __init__(self, client: PoliteClient):
        self.client = client
        self._form_state: dict[str, str] | None = None

    def list_month(self, year: int, month: int) -> list[ListingEntry]:
        """Notifications issued in a month (month=0 means the whole year)."""
        if self._form_state is None:
            self._form_state = parse_form_state(self.client.get_text(LISTING_URL))
        data = {**self._form_state, "hdnYear": str(year), "hdnMonth": str(month)}
        return parse_listing(self.client.post_text(LISTING_URL, data))

    def list_master_directions(self) -> list[ListingEntry]:
        return parse_listing(self.client.get_text(MD_INDEX_URL))

    def fetch_detail(self, rbi_id: int, refresh: bool = False) -> tuple[str, Path | None]:
        key = f"rbi/notifications/{rbi_id}.html"
        html = self.client.get_text(detail_url(rbi_id), cache_key=key, refresh=refresh)
        path = self.client.cache_dir / key if self.client.cache_dir else None
        return html, path
