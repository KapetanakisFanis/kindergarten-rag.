"""
Εργασία 2 - Φύλλο βαθμολόγησης (Excel) με την κλίμακα της εργασίας.

Δύο ξεχωριστές βαθμολογήσεις για κάθε απάντηση:
  1) ΟΡΘΟΤΗΤΑ: σωστό / μερικώς σωστό / λάθος / επινοημένο
  2) ΠΑΡΑΠΟΜΠΗ (μόνο με RAG): στηρίζει / εν μέρει / δεν στηρίζει / Ν/Α
     = στηρίζει πραγματικά το απόσπασμα αυτό που λέει η απάντηση;

Το φύλλο «Σύνοψη» υπολογίζεται με τύπους καθώς συμπληρώνεις.

Χρήση:
  python src/make_scoring_sheet.py
  python src/make_scoring_sheet.py --rag results/E2.jsonl --norag results/E2_norag.jsonl

Απαιτεί: pip install openpyxl
"""
import argparse
import csv
import json
import re
import unicodedata

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

FONT = "Arial"
YELLOW = PatternFill("solid", fgColor="FFF2CC")
HEAD = PatternFill("solid", fgColor="D9E1F2")
GREY = PatternFill("solid", fgColor="E7E6E6")
THIN = Side(style="thin", color="BBBBBB")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

CORRECT = ["σωστό", "μερικώς σωστό", "λάθος", "επινοημένο"]
CITE = ["στηρίζει", "εν μέρει", "δεν στηρίζει", "Ν/Α"]
GROUPS = ["Με κάλυψη", "Άρνηση"]
SUGGEST_RE = re.compile(r"(Δραστηριότητα που (προτείνεται|μπορεί να προταθεί)|Προτεινόμενη δραστηριότητα)", re.I)
REFUSAL_START = "Το υλικό του Προγράμματος Σπουδών που διαθέτω δεν καλύπτει αυτή την ερώτηση"

ERROR_CODES = [
    ("ΟΚ", "κανένα σφάλμα"),
    ("ΑΝΑ", "η ανάκτηση απέτυχε: το σχετικό υλικό δεν έφτασε στο μοντέλο"),
    ("ΑΓΝ", "το μοντέλο αγνόησε σχετικό υλικό που είχε ανακτηθεί"),
    ("ΕΠΙΝ", "επινοημένο περιεχόμενο (δραστηριότητα ή λεπτομέρεια που δεν υπάρχει στο υλικό)"),
    ("ΠΑΡΑ", "παραμόρφωση της πηγής (αλλαγμένη λέξη, αντίστροφο νόημα, προσαρμογή στην ερώτηση)"),
    ("ΣΥΝ", "ψευδής σύνδεση δραστηριοτήτων με συγκεκριμένο μαθησιακό αποτέλεσμα"),
    ("ΨΑΡ", "ψευδής άρνηση: είπε ότι δεν καλύπτεται ενώ υπάρχει υλικό"),
    ("ΛΑΡ", "απάντησε ενώ έπρεπε να αρνηθεί (ερώτηση εκτός υλικού ή εσφαλμένη υπόθεση)"),
    ("ΕΛΛ", "ελλιπής απάντηση (σωστή αλλά χωρίς βασικό στοιχείο)"),
    ("ΠΑΡΜ", "η παραπομπή υπάρχει αλλά το απόσπασμα δεν στηρίζει τον ισχυρισμό"),
]

# στήλες της «Βαθμολόγησης»
HEADERS = ["id", "ομάδα", "τύπος", "ερώτηση", "αναμενόμενη συμπεριφορά", "τι περιέχει το υλικό (πρόχειρο gold)",
           "απάντηση ΜΕ RAG", "πηγές που αναφέρονται", "Αυτόματοι έλεγχοι", "απάντηση ΧΩΡΙΣ RAG",
           "RAG: ορθότητα", "RAG: παραπομπή", "RAG: κωδικός σφάλματος", "Χωρίς RAG: ορθότητα", "σχόλια", "κατηγορία",
           "Claude: RAG ορθότητα", "Claude: RAG παραπομπή", "Claude: RAG κωδικός", "Claude: χωρίς RAG ορθότητα",
           "Claude: σχόλιο (RAG)", "Claude: σχόλιο (χωρίς RAG)"]
WIDTHS = [5, 9, 16, 38, 20, 48, 70, 40, 44, 55, 14, 14, 14, 14, 40, 14, 14, 14, 14, 14, 60, 45]


def norm(s):
    return unicodedata.normalize("NFC", s or "")


def is_real_refusal(answer):
    a = norm(answer).strip()
    return a.startswith(norm(REFUSAL_START)) and len(a) < len(REFUSAL_START) + 40


def load_jsonl(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        print(f"⚠ Δεν βρέθηκε το {path}")
        return []


def gold_recall(prompt, phrases):
    blob = norm(prompt)
    missing = [p for p in phrases if norm(p) not in blob]
    return len(phrases) - len(missing), len(phrases), missing


def auto_checks(q, rag, gold_map):
    """Αυτόματες ενδείξεις (βοήθημα, όχι βαθμολογία)."""
    if rag is None:
        return "—"
    notes = []
    ans = rag.get("answer", "")
    refused = is_real_refusal(ans)
    if q["αναμενόμενη_συμπεριφορά"].startswith("Άρνηση"):
        notes.append("✔ αρνήθηκε όπως αναμενόταν" if refused else "✘ ΔΕΝ αρνήθηκε (αναμενόταν άρνηση ή ρητή επισήμανση)")
    elif refused:
        notes.append("✘ αρνήθηκε ενώ αναμενόταν απάντηση (πιθανή ψευδής άρνηση)")
    if rag.get("invalid_citations"):
        notes.append(f"✘ άκυροι αριθμοί παραπομπής: {rag['invalid_citations']}")
    if rag.get("no_citations") and not refused:
        notes.append("⚠ δεν περιέχει παραπομπές")
    if SUGGEST_RE.search(ans):
        notes.append("⚠ περιέχει «προτεινόμενη δραστηριότητα»: έλεγξε αν υπάρχει στο υλικό")
    if "δεν καλύπτ" in ans and not refused:
        notes.append("⚠ λέει «δεν καλύπτεται» για μέρος της ερώτησης")
    g = gold_map.get(norm(q["ερώτηση"]))
    if g:
        n, tot, missing = gold_recall(rag.get("prompt", ""), g)
        notes.append(f"Ανάκτηση: {n}/{tot} φράσεις-στόχοι έφτασαν στο μοντέλο"
                     + (f" (λείπουν: {'; '.join(missing)})" if missing else ""))
    return "\n".join(notes) if notes else "—"


def style(c, bold=False, fill=None, wrap=True, size=10):
    c.font = Font(name=FONT, size=size, bold=bold)
    c.alignment = Alignment(wrap_text=wrap, vertical="top")
    c.border = BORDER
    if fill:
        c.fill = fill


def write_instructions(ws):
    lines = [
        ("Φύλλο βαθμολόγησης - Εργασία 2", True),
        ("Συμπλήρωσε ΜΟΝΟ τα κίτρινα κελιά του φύλλου «Βαθμολόγηση» (στήλες K έως O). Η «Σύνοψη» υπολογίζεται μόνη της.", False),
        ("", False),
        ("1) Ορθότητα της απάντησης (στήλες K και N: επιλογή από λίστα)", True),
        ("σωστό = όλοι οι ισχυρισμοί στηρίζονται στο Πρόγραμμα Σπουδών και η απάντηση είναι ουσιαστικά πλήρης. Για ερώτηση χωρίς κάλυψη: σωστή άρνηση.", False),
        ("μερικώς σωστό = χρήσιμη απάντηση που είναι ελλιπής ή περιέχει έναν ισχυρισμό που δεν στηρίζεται.", False),
        ("λάθος = λανθασμένο περιεχόμενο ή ψευδής δήλωση ότι το υλικό δεν καλύπτει την ερώτηση.", False),
        ("επινοημένο = το κεντρικό περιεχόμενο (δραστηριότητα, αριθμός, στόχος) δεν υπάρχει στο Πρόγραμμα και παρουσιάζεται ως υπαρκτό ή ως προερχόμενο από αυτό.", False),
        ("", False),
        ("2) Παραπομπή (στήλη L, μόνο για απαντήσεις με RAG)", True),
        ("Ερώτηση: στηρίζει πραγματικά το απόσπασμα αυτό που λέει η απάντηση; Άνοιξε το PDF στη σελίδα που αναφέρεται και διάβασε το κείμενο. Μια απάντηση μπορεί να είναι σωστή και η παραπομπή της λάθος.", False),
        ("στηρίζει = κάθε παραπομπή υπάρχει ΚΑΙ το απόσπασμα στηρίζει τον ισχυρισμό δίπλα της.", False),
        ("εν μέρει = μερικές παραπομπές δεν στηρίζουν τον ισχυρισμό ή λείπουν από ουσιώδεις ισχυρισμούς.", False),
        ("δεν στηρίζει = οι παραπομπές δεν στηρίζουν τους ισχυρισμούς ή είναι άκυρες.", False),
        ("Ν/Α = η απάντηση είναι άρνηση και δεν έχει ισχυρισμούς προς παραπομπή.", False),
        ("", False),
        ("3) Κωδικός σφάλματος (στήλη M: επιλογή από λίστα ή γράψε περισσότερους, π.χ. ΕΠΙΝ+ΑΝΑ)", True),
    ]
    lines += [(f"{code} = {desc}", False) for code, desc in ERROR_CODES]
    lines += [
        ("", False),
        ("Παράδειγμα συμπλήρωσης (ερώτηση 8, βέβαιο/πιθανό/αδύνατο)", True),
        ("RAG ορθότητα = επινοημένο · παραπομπή = εν μέρει · κωδικός = ΕΠΙΝ+ΑΝΑ · σχόλιο = «Η δραστηριότητα με τα τετράγωνα δεν υπάρχει στο κείμενο. Η δραστηριότητα με τα αυτοκόλλητα (σ.72) δεν ανακτήθηκε.»", False),
        ("", False),
        ("Οδηγίες", True),
        ("Βαθμολόγησε πρώτα χωρίς να κοιτάς τη στήλη «Αυτόματοι έλεγχοι», και μετά σύγκρινε. Οι αυτόματοι έλεγχοι είναι βοήθημα και όχι βαθμολογία.", False),
        ("Οι στήλες Q–V («Claude») είναι ΚΡΥΜΜΕΝΕΣ και περιέχουν την προσωρινή βαθμολογία δεύτερου αξιολογητή (εργαλείο AI). Άνοιξέ τες (κουμπί + πάνω από τις στήλες) ΜΟΝΟ αφού βαθμολογήσεις εσύ. Η τελική βαθμολογία είναι δική σου απόφαση.", False),
        ("Στη βαθμολογία «χωρίς RAG» κρίνεται αν η απάντηση συμφωνεί με το Πρόγραμμα και αν του αποδίδει κάτι που δεν περιέχει. Μια γενική, εύλογη αλλά ανεξάρτητη από το Πρόγραμμα απάντηση είναι «μερικώς σωστό».", False),
        ("Οι σημειώσεις «τι περιέχει το υλικό» είναι προσχέδιο: επαλήθευσέ τες στο PDF όπου χρειάζεται.", False),
    ]
    for i, (text, bold) in enumerate(lines, 1):
        c = ws.cell(row=i, column=1, value=text)
        c.font = Font(name=FONT, size=12 if i == 1 else 10, bold=bold)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 135


def write_summary(ws_sum, last):
    S = "'Βαθμολόγηση'!"
    rng = lambda col: f"{S}${col}$2:${col}${last}"
    ws_sum["A1"] = "Σύνοψη βαθμολόγησης (υπολογίζεται αυτόματα)"
    ws_sum["A1"].font = Font(name=FONT, size=12, bold=True)

    row = 3

    def header(title, cols):
        nonlocal row
        style(ws_sum.cell(row=row, column=1, value=title), bold=True, fill=HEAD)
        for j, h in enumerate(cols, 2):
            style(ws_sum.cell(row=row, column=j, value=h), bold=True, fill=HEAD)
        row += 1

    def put(label, formulas, fmt="0"):
        nonlocal row
        style(ws_sum.cell(row=row, column=1, value=label))
        for j, f in enumerate(formulas, 2):
            c = ws_sum.cell(row=row, column=j, value=f)
            style(c, bold=True)
            c.number_format = fmt
        row += 1

    # --- ορθότητα: 4 στήλες (RAG εσύ, RAG Claude, χωρίς RAG εσύ, χωρίς RAG Claude)
    cols = [("K", "Με RAG: εσύ"), ("Q", "Με RAG: Claude"), ("N", "Χωρίς RAG: εσύ"), ("T", "Χωρίς RAG: Claude")]
    for title, group in (("Ορθότητα, όλες οι ερωτήσεις", None),
                         ("Ορθότητα, ερωτήσεις με κάλυψη στο υλικό", "Με κάλυψη"),
                         ("Ορθότητα, ερωτήσεις προς άρνηση (εκτός υλικού ή εσφαλμένη υπόθεση)", "Άρνηση")):
        header(title, [h for _, h in cols])
        first = row
        for cat in CORRECT:
            fs = []
            for col, _ in cols:
                if group is None:
                    fs.append(f'=COUNTIF({rng(col)},"{cat}")')
                else:
                    fs.append(f'=COUNTIFS({rng(col)},"{cat}",{rng("P")},"{group}")')
            put(cat, fs)
        put("Σύνολο βαθμολογημένων", [f"=SUM({get_column_letter(j)}{first}:{get_column_letter(j)}{row - 1})" for j in range(2, 6)])
        row += 1

    # --- παραπομπές
    header("Παραπομπή (μόνο απαντήσεις με RAG)", ["Εσύ", "Claude"])
    first = row
    for cat in CITE:
        put(cat, [f'=COUNTIF({rng("L")},"{cat}")', f'=COUNTIF({rng("R")},"{cat}")'])
    put("Σύνολο βαθμολογημένων", [f"=SUM(B{first}:B{row - 1})", f"=SUM(C{first}:C{row - 1})"])
    row += 1

    # --- συμφωνία
    header("Συμφωνία με Claude (ποσοστό ίδιας βαθμολογίας στις ερωτήσεις που βαθμολόγησες)", ["Ποσοστό"])
    for label, a, b in (("Ορθότητα με RAG", "K", "Q"), ("Παραπομπή", "L", "R"), ("Ορθότητα χωρίς RAG", "N", "T")):
        put(label, [f'=IFERROR(SUMPRODUCT(({rng(a)}<>"")*({rng(a)}={rng(b)}))/COUNTA({rng(a)}),"—")'], fmt="0%")
    row += 1

    # --- κωδικοί σφαλμάτων
    header("Κωδικοί σφαλμάτων (απαντήσεις με RAG)", ["Εσύ", "Claude"])
    for code, desc in ERROR_CODES:
        put(f"{code}: {desc}", [f'=COUNTIF({rng("M")},"*{code}*")', f'=COUNTIF({rng("S")},"*{code}*")'])
    ws_sum.cell(row=row + 1, column=1,
                value="Ο κωδικός μετριέται όπου εμφανίζεται, ακόμη κι αν είναι γραμμένος μαζί με άλλον (π.χ. ΕΠΙΝ+ΑΝΑ).").font = Font(name=FONT, size=9, italic=True)
    ws_sum.column_dimensions["A"].width = 90
    for col in "BCDE":
        ws_sum.column_dimensions[col].width = 18


def build(args):
    questions = list(csv.DictReader(open(args.questions, encoding="utf-8-sig", newline="")))
    rag, norag = load_jsonl(args.rag), load_jsonl(args.norag)
    gold = {}
    try:
        for g in json.load(open(args.gold, encoding="utf-8")):
            gold[norm(g["question"])] = g["gold"]
    except FileNotFoundError:
        pass
    claude = {}
    try:
        for c in json.load(open(args.claude, encoding="utf-8")):
            claude[int(c["id"])] = c
    except FileNotFoundError:
        pass

    wb = Workbook()
    ws0 = wb.active
    ws0.title = "Οδηγίες"
    write_instructions(ws0)

    ws = wb.create_sheet("Βαθμολόγηση")
    for j, h in enumerate(HEADERS, 1):
        style(ws.cell(row=1, column=j, value=h), bold=True, fill=(GREY if j >= 17 else HEAD))
        ws.column_dimensions[get_column_letter(j)].width = WIDTHS[j - 1]

    for i, q in enumerate(questions):
        r = i + 2
        qtext = q["ερώτηση"]
        rr = rag[i] if i < len(rag) and norm(rag[i]["question"]) == norm(qtext) else None
        nr = norag[i] if i < len(norag) and norm(norag[i]["question"]) == norm(qtext) else None
        if i < len(rag) and rr is None:
            print(f"⚠ Η ερώτηση {q['id']} δεν ταιριάζει με το αποτέλεσμα RAG στη θέση {i + 1}")
        sources = "\n".join(f"[{c['n']}] {c['source']}" for c in (rr or {}).get("cited", []))
        group = "Άρνηση" if q["αναμενόμενη_συμπεριφορά"].startswith("Άρνηση") else "Με κάλυψη"
        cl = claude.get(int(q["id"]))
        vals = [int(q["id"]), q["ομάδα"], q["τύπος"], qtext, q["αναμενόμενη_συμπεριφορά"], q["τι_περιέχει_το_υλικό"],
                (rr or {}).get("answer", ""), sources, auto_checks(q, rr, gold), (nr or {}).get("answer", ""),
                None, None, None, None, None, group]
        vals += ([cl["rag_correct"], cl["rag_cite"], cl["rag_code"], cl["norag_correct"], cl["comment_rag"], cl["comment_norag"]]
                 if cl else [None] * 6)
        for j, v in enumerate(vals, 1):
            c = ws.cell(row=r, column=j, value=v)
            style(c, fill=YELLOW if 11 <= j <= 15 else (GREY if j >= 17 else None))
        ws.row_dimensions[r].height = 260
    last = len(questions) + 1
    ws.freeze_panes = "E2"
    if claude:
        ws.column_dimensions.group("Q", "V", hidden=True)

    dv_correct = DataValidation(type="list", formula1='"' + ",".join(CORRECT) + '"', allow_blank=True)
    dv_cite = DataValidation(type="list", formula1='"' + ",".join(CITE) + '"', allow_blank=True)
    dv_code = DataValidation(type="list", formula1='"' + ",".join(c for c, _ in ERROR_CODES) + '"',
                             allow_blank=True, showErrorMessage=False)
    for dv, col in ((dv_correct, "K"), (dv_cite, "L"), (dv_code, "M"), (dv_correct, "N")):
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{last}")

    write_summary(wb.create_sheet("Σύνοψη"), last)
    wb.active = wb.sheetnames.index("Βαθμολόγηση")
    wb.save(args.out)
    print(f"Αποθηκεύτηκε: {args.out}  ({len(questions)} ερωτήσεις, RAG: {len(rag)}, χωρίς RAG: {len(norag)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions_draft.csv")
    ap.add_argument("--rag", default="results/E2.jsonl")
    ap.add_argument("--norag", default="results/E2_norag.jsonl")
    ap.add_argument("--gold", default="eval/retrieval_gold.json")
    ap.add_argument("--claude", default="eval/claude_scores.json")
    ap.add_argument("--out", default="results/scoring_sheet.xlsx")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
