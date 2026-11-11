"""
Βήμα 1 - Εξαγωγή κειμένου από το Πρόγραμμα Σπουδών και δημιουργία chunks.

Τι κάνει:
  1. Διαβάζει συγκεκριμένα εύρη σελίδων του PDF του ΙΕΠ (Γλώσσα, Μαθηματικά).
  2. Στις σελίδες-πίνακες διαβάζει ΑΝΑ ΣΤΗΛΗ (υποενότητες | μαθησιακά αποτελέσματα |
     δραστηριότητες), γιατί ο πίνακας απλώνεται σε πολλές σελίδες.
  3. Σπάει κάθε στήλη σε κουκκίδες και κρατά για κάθε κουκκίδα: υποενότητα,
     κατηγορία (Γνώσεις/Δεξιότητες/Στάσεις), σελίδα.
  4. Ενώνει γειτονικές κουκκίδες μέχρι MAX_CHARS χαρακτήρες (το "μέγεθος chunk").
  5. Γράφει data/processed/chunks.jsonl (ένα JSON ανά γραμμή).

Χρήση:
  python src/ingest.py --pdf data/raw/programma_spoudon.pdf --max-chars 700
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import pdfplumber

# ----------------------------------------------------------------------------
# Ρυθμίσεις: ποιες σελίδες χρησιμοποιούμε. Οι αριθμοί σελίδας του PDF
# ταυτίζονται με τους τυπωμένους αριθμούς.
# ----------------------------------------------------------------------------
SECTIONS = [
    {
        "name": "Γλώσσα",
        "code": "Α.1",
        "pages": range(28, 38),        # 28-37
        "intro_pages": [28, 29],       # σελίδες με συνεχές κείμενο (εισαγωγή)
        "start_marker": "Α.1 Γλώσσα",  # στη σελ. 28 κόβουμε ό,τι είναι πάνω από αυτό
    },
    {
        "name": "Μαθηματικά",
        "code": "Γ.1",
        "pages": range(62, 74),        # 62-73
        "intro_pages": [62],
        "start_marker": "Γ.1 Μαθηματικά",
    },
]

SOURCE = "Πρόγραμμα Σπουδών Προσχολικής Εκπαίδευσης (Διευρυμένη Εκδοχή, ΙΕΠ 2022)"
CATEGORIES = {"Γνώσεις", "Δεξιότητες", "Στάσεις"}
BULLET_CHARS = ("\uf0b7", "•")
SUBUNIT_RE = re.compile(r"^([Α-Δ]\.\d+\.\d+)\s+(.*)$")
TOPIC_SPLIT_RE = re.compile(r"(?:^|\s)(i{1,3}|iv|v|vi{0,3})\.\s")


# ----------------------------------------------------------------------------
# Βοηθητικές συναρτήσεις
# ----------------------------------------------------------------------------
def clamp_bbox(page, bbox):
    """Κρατά το bbox μέσα στα όρια της σελίδας (αλλιώς το crop βγάζει σφάλμα)."""
    x0, top, x1, bottom = bbox
    px0, ptop, px1, pbottom = page.bbox
    return (max(x0, px0), max(top, ptop), min(x1, px1), min(bottom, pbottom))


def content_band(page):
    """Επιστρέφει (top, bottom) του κυρίως κειμένου, χωρίς header/footer."""
    top, bottom = page.bbox[1], page.bbox[3]
    for m in page.search(r"Προσχολική Εκπαίδευση"):
        if m["top"] < 80:               # η κεφαλίδα της σελίδας
            top = max(top, m["bottom"] + 2)
    for m in page.search(r"Επιχειρησιακό Πρόγραμμα"):
        if m["top"] > page.height / 2:  # το footer ΕΣΠΑ
            bottom = min(bottom, m["top"] - 2)
    return top, bottom


def clean_lines(text):
    """Σπάει σε γραμμές, καθαρίζει κουκκίδες και κενά."""
    out = []
    for line in (text or "").splitlines():
        # NFC: το PDF αποθηκεύει πολλούς τόνους ως ξεχωριστό χαρακτήρα (υ + U+0301).
        # Χωρίς κανονικοποίηση, οι αναζητήσεις κειμένου αποτυγχάνουν και ο tokenizer
        # του γλωσσικού μοντέλου βλέπει «σπασμένες» λέξεις.
        line = unicodedata.normalize("NFC", line.replace("\uf0b7", "•")).strip()
        if line:
            out.append(line)
    return out


def join_lines(lines):
    """Ενώνει γραμμές σε κείμενο. Το 'περήφανοι/-' + 'ες' ενώνεται χωρίς κενό."""
    text = ""
    for ln in lines:
        if text.endswith("/-"):
            text += ln
        else:
            text += (" " if text else "") + ln
    return re.sub(r"\s+", " ", text).strip()


MARKER_RE = re.compile(r"\((i{1,3}|iv|v|vi{0,3})\)")


def annotate_topics(text, topics):
    """Αντικαθιστά το σήμα '(ii)' στο τέλος μιας κουκκίδας με '(ii: Μεταγλωσσική επίγνωση)'.
    Έτσι το νόημα του σήματος γίνεται μέρος του κειμένου που βλέπει το μοντέλο
    (και το embedding), αντί να μένει ένα ακατανόητο (ii)."""
    return MARKER_RE.sub(lambda m: f"({m.group(1)}: {topics[m.group(1)]})"
                         if m.group(1) in topics else m.group(0), text)


def parse_topics(sub_text):
    """'Α.1.1 Προφορική Επικοινωνία i. Κατανόηση ... ii. Μεταγλωσσική ...'
    -> code, title, {'i': 'Κατανόηση ...', 'ii': '...'}"""
    flat = join_lines(sub_text.splitlines())
    m = SUBUNIT_RE.match(flat)
    if not m:
        return None, None, {}
    code, rest = m.group(1), m.group(2)
    parts = TOPIC_SPLIT_RE.split(" " + rest)
    title = parts[0].strip()
    topics = {}
    for i in range(1, len(parts) - 1, 2):
        topics[parts[i]] = parts[i + 1].strip()
    return code, title, topics


# ----------------------------------------------------------------------------
# Εξαγωγή: σελίδες-πίνακες
# ----------------------------------------------------------------------------
def extract_table_columns(page):
    """Για μια σελίδα-πίνακα επιστρέφει λίστα γραμμών πίνακα:
    [(sub_text, outcomes_text, activities_text), ...] ή []"""
    tables = page.find_tables()
    if not tables:
        return []
    main = max(tables, key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))
    rows = []
    for row in main.rows:
        cells = [c for c in row.cells if c is not None]
        if len(cells) < 3:
            continue
        sub_bb, out_bb, act_bb = cells[-3:]   # οι 3 τελευταίες στήλες
        texts = [page.crop(clamp_bbox(page, bb)).extract_text() or ""
                 for bb in (sub_bb, out_bb, act_bb)]
        joined = " ".join(texts)
        # παράλειψη γραμμών-επικεφαλίδων του πίνακα
        if "Προσδοκώμενα" in joined or "Οι μαθητές/-τριες:" in joined \
                or "Οι μαθητές/-τριες να είναι" in joined:
            continue
        rows.append(tuple(texts))
    return rows


def tagged_lines(page_no, text):
    return [(page_no, ln) for ln in clean_lines(text)]


def parse_bullets(tagged):
    """tagged: [(σελίδα, γραμμή)] -> [{'category','text','pages'}]
    Μια κουκκίδα ξεκινά με • και συνεχίζεται μέχρι την επόμενη κουκκίδα ή
    επικεφαλίδα κατηγορίας. Μπορεί να συνεχίζεται και στην επόμενη σελίδα."""
    bullets, cur, category = [], None, None
    for page_no, ln in tagged:
        if ln in CATEGORIES:
            category = ln
            cur = None
            continue
        if ln.startswith(BULLET_CHARS):
            cur = {"category": category, "lines": [ln.lstrip("•").strip()], "pages": {page_no}}
            bullets.append(cur)
        elif cur is not None:
            cur["lines"].append(ln)
            cur["pages"].add(page_no)
    out = []
    for b in bullets:
        text = join_lines(b["lines"])
        if text:
            out.append({"category": b["category"], "text": text, "pages": sorted(b["pages"])})
    return out


# ----------------------------------------------------------------------------
# Εξαγωγή: σελίδες με συνεχές κείμενο (εισαγωγές)
# ----------------------------------------------------------------------------
def extract_prose(page, start_marker=None):
    top, bottom = content_band(page)
    if start_marker:
        hits = page.search(re.escape(start_marker))
        if hits:
            top = max(top, hits[0]["bottom"] + 1)
    box = clamp_bbox(page, (page.bbox[0], top, page.bbox[2], bottom))
    return join_lines(clean_lines(page.crop(box).extract_text()))


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.;·])\s+", text) if s.strip()]


def pack(pieces, max_chars, sep=" "):
    """Συσσωρεύει κομμάτια μέχρι max_chars, χωρίς να κόβει κομμάτι στη μέση."""
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(sep) + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + sep + p) if cur else p
    if cur:
        chunks.append(cur)
    return chunks


# ----------------------------------------------------------------------------
# Κύρια ροή
# ----------------------------------------------------------------------------
def page_label(pages):
    return f"σ. {pages[0]}" if len(pages) == 1 else f"σ. {pages[0]}–{pages[-1]}"


def group_by_category(bullets):
    """Ομαδοποιεί διαδοχικές κουκκίδες που έχουν την ίδια κατηγορία."""
    groups, group = [], []
    for b in bullets:
        if group and group[-1]["category"] != b["category"]:
            groups.append(group)
            group = []
        group.append(b)
    if group:
        groups.append(group)
    return groups


def pack_bullets(group, max_chars, per_bullet=False):
    """Ενώνει κουκκίδες σε chunks ≤ max_chars. Η κουκκίδα δεν κόβεται ποτέ.
    Με per_bullet=True κάθε κουκκίδα γίνεται δικό της chunk."""
    if per_bullet:
        return [("- " + b["text"], list(b["pages"])) for b in group]
    packed, items, pages = [], [], []
    for b in group:
        line = "- " + b["text"]
        if items and len("\n".join(items)) + len(line) + 1 > max_chars:
            packed.append(("\n".join(items), pages))
            items, pages = [], []
        items.append(line)
        pages += b["pages"]
    if items:
        packed.append(("\n".join(items), pages))
    return packed


def build_chunks(pdf_path, max_chars, activity_per_bullet=False, embed_mode="full"):
    chunks = []

    def add(section, header, body, pages, **meta):
        pages = sorted(set(pages))
        chunks.append({
            "id": f"c{len(chunks):04d}",
            "section": section,
            "header": header,
            "text": f"{header}\n{body}",   # αυτό βλέπει το γλωσσικό μοντέλο
            # το κείμενο που γίνεται embedding: 'full' = επικεφαλίδα+σώμα, 'body' = μόνο σώμα
            "embed_text": body if embed_mode == "body" else f"{header}\n{body}",
            "body": body,
            "page_start": pages[0],
            "page_end": pages[-1],
            "citation": f"{SOURCE}, {header}, {page_label(pages)}",
            **meta,
        })

    with pdfplumber.open(pdf_path) as pdf:
        for sec in SECTIONS:
            name = sec["name"]

            # (1) εισαγωγικό κείμενο της ενότητας
            for i, pno in enumerate(sec["intro_pages"]):
                marker = sec["start_marker"] if i == 0 else None
                text = extract_prose(pdf.pages[pno - 1], marker)
                for piece in pack(split_sentences(text), max_chars):
                    add(name, f"{name} › Εισαγωγή", piece, [pno], kind="intro",
                        subunit_code=sec["code"], subunit=name, category=None, topics=[])

            # (2) πίνακας: συλλογή γραμμών ανά στήλη, κατά υποενότητα
            state, order, current = {}, [], None
            for pno in sec["pages"]:
                if pno in sec["intro_pages"]:
                    continue
                for sub_t, out_t, act_t in extract_table_columns(pdf.pages[pno - 1]):
                    code, title, topics = parse_topics(sub_t) if sub_t.strip() else (None, None, {})
                    if code:
                        current = code
                        state[current] = {"title": title, "topics": topics, "out": [], "act": []}
                        order.append(current)
                    elif sub_t.strip() and current:
                        # η στήλη υποενοτήτων συνεχίζεται στην επόμενη σελίδα
                        _, _, more = parse_topics("Χ.0.0 χ " + sub_t)
                        state[current]["topics"].update(more)
                    if current is None:
                        continue
                    state[current]["out"] += tagged_lines(pno, out_t)
                    state[current]["act"] += tagged_lines(pno, act_t)

            # (3) κουκκίδες -> chunks
            for code in order:
                s = state[code]
                header_base = f"{name} › {code} {s['title']}"
                for kind, label, lines in (
                    ("outcome", "Προσδοκώμενα μαθησιακά αποτελέσματα", s["out"]),
                    ("activity", "Ενδεικτικές δραστηριότητες", s["act"]),
                ):
                    for group in group_by_category(parse_bullets(lines)):
                        cat = group[0]["category"]
                        for b in group:
                            b["text"] = annotate_topics(b["text"], s["topics"])
                        for body, pages in pack_bullets(group, max_chars,
                                                        per_bullet=(kind == "activity" and activity_per_bullet)):
                            header = f"{header_base} › {label}" + (f" › {cat}" if cat else "")
                            # τα (i), (ii) στο τέλος κάθε αποτελέσματος παραπέμπουν σε θέματα
                            refs = sorted(set(re.findall(r"\((i{1,3}|iv|v|vi{0,3})[:)]", body)))
                            topic_names = [f"{r}. {s['topics'][r]}" for r in refs if r in s["topics"]]
                            add(name, header, body, pages, kind=kind, subunit_code=code,
                                subunit=s["title"], category=cat, topics=topic_names)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/raw/programma_spoudon.pdf")
    ap.add_argument("--out", default="data/processed/chunks.jsonl")
    ap.add_argument("--max-chars", type=int, default=700,
                    help="μέγιστο μέγεθος chunk σε χαρακτήρες (παράμετρος του chunking)")
    ap.add_argument("--activity-per-bullet", action="store_true",
                    help="κάθε ενδεικτική δραστηριότητα γίνεται ξεχωριστό chunk")
    ap.add_argument("--embed-mode", choices=["full", "body"], default="full",
                    help="τι κειμένο γίνεται embedding: επικεφαλίδα+σώμα (full) ή μόνο σώμα (body)")
    args = ap.parse_args()

    chunks = build_chunks(args.pdf, args.max_chars, args.activity_per_bullet, args.embed_mode)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = [len(c["body"]) for c in chunks]
    print(f"Γράφτηκαν {len(chunks)} chunks -> {args.out}")
    print(f"Μέγεθος (χαρακτήρες): min={min(sizes)}, μέσος={sum(sizes) // len(sizes)}, max={max(sizes)}")
    for kind in ("intro", "outcome", "activity"):
        print(f"  {kind}: {sum(c['kind'] == kind for c in chunks)}")


if __name__ == "__main__":
    main()
