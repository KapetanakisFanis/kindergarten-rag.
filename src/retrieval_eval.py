"""
Μέτρηση ανάκτησης με φράσεις-στόχους (gold). Δεν χρειάζεται το γλωσσικό μοντέλο.

Για κάθε ερώτηση του eval/retrieval_gold.json ελέγχουμε ποιες από τις φράσεις-στόχους
(κείμενο συγκεκριμένων αποτελεσμάτων/δραστηριοτήτων που απαντούν στην ερώτηση)
υπάρχουν στα αποσπάσματα που ανακτήθηκαν. Είναι αυστηρότερο από το retrieval_check.py.

Οι ερωτήσεις χωρίζονται σε 'dev' (1-10, με τις οποίες έγινε η ρύθμιση) και 'test'
(11-23, που δεν χρησιμοποιήθηκαν για ρυθμίσεις). Η επιλογή ρύθμισης γίνεται στο dev
και η αναφορά αποτελεσμάτων στο test.

Χρήση:
  python src/retrieval_eval.py --index-dir data/index_v5 -v
  python src/retrieval_eval.py --index-dir data/index_v5 --k-activity 6 --label "k_act=6"
"""
import argparse
import json
import unicodedata

from common import INDEX_DIR, MODEL_NAME


def norm(s):
    return unicodedata.normalize("NFC", s)


def score_question(retrieved_texts, gold_phrases):
    """Επιστρέφει (recall, λίστα φράσεων που λείπουν)."""
    blob = norm(" ".join(retrieved_texts))
    missing = [p for p in gold_phrases if norm(p) not in blob]
    return 1 - len(missing) / len(gold_phrases), missing


def summarize(rows):
    if not rows:
        return "—"
    recall = sum(r for r, _ in rows) / len(rows)
    full = sum(1 for _, m in rows if not m)
    return f"recall={recall:.2f}, πλήρης κάλυψη={full}/{len(rows)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/retrieval_gold.json")
    ap.add_argument("--index-dir", default=INDEX_DIR)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--k-outcome", type=int, default=3)
    ap.add_argument("--k-activity", type=int, default=3)
    ap.add_argument("--k-intro", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    from retrieve import Retriever
    r = Retriever(args.model, args.index_dir)
    gold = json.load(open(args.gold, encoding="utf-8"))
    by_split = {"dev": [], "test": []}
    for g in gold:
        hits = r.search_balanced(g["question"], args.k_outcome, args.k_activity, args.k_intro)
        recall, missing = score_question([h["text"] for h in hits], g["gold"])
        by_split[g["split"]].append((recall, missing))
        if args.verbose:
            mark = "OK " if not missing else "ΛΕΙΠΟΥΝ"
            print(f"[{g['split']}] Ε{g['id']:>2} recall={recall:.2f} {mark} {g['question']}")
            for m in missing:
                print(f"        λείπει: {m}")
    label = args.label or f"{args.index_dir} (3/{args.k_activity})"
    print(f"{label:<42} | dev: {summarize(by_split['dev'])} | test: {summarize(by_split['test'])}")


if __name__ == "__main__":
    main()
