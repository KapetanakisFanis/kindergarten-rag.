"""
Βήμα 3 - Σύστημα RAG: ανάκτηση αποσπασμάτων -> prompt -> Krikri -> απάντηση με παραπομπές.

Χρήση:
  python src/rag.py "Πώς μπορώ να δουλέψω τη φωνολογική επίγνωση;"
  python src/rag.py --interactive
  python src/rag.py --no-rag "ερώτηση..."                 # χωρίς ανάκτηση (βάση σύγκρισης, Εργασία 2)
  python src/rag.py --batch eval/test_questions.txt       # μία ερώτηση ανά γραμμή

Οι παραπομπές ΔΕΝ γράφονται από το μοντέλο. Το μοντέλο βάζει μόνο αριθμούς [1], [2]...
και ο κώδικας τους αντιστοιχίζει στα πραγματικά αποσπάσματα (υποενότητα, σελίδα).
Αριθμοί που δεν αντιστοιχούν σε απόσπασμα επισημαίνονται ως άκυροι.
"""
import argparse
import json
import re
from pathlib import Path

from common import INDEX_DIR, MODEL_NAME

LLM_PATH = "models/llama-krikri-8b-instruct-q4_k_m.gguf"
REFUSAL = "Το υλικό του Προγράμματος Σπουδών που διαθέτω δεν καλύπτει αυτή την ερώτηση."

SYSTEM_RAG = f"""Είσαι βοηθός για νηπιαγωγούς στην Ελλάδα. Απαντάς στα ελληνικά, σύντομα και πρακτικά.
Κανόνες:
1. Χρησιμοποίησε ΜΟΝΟ πληροφορίες από τα αποσπάσματα του Προγράμματος Σπουδών που σου δίνονται. Μην προσθέτεις γνώσεις από αλλού.
2. Διάβασε ΟΛΑ τα αποσπάσματα. Η ερώτηση μπορεί να χρησιμοποιεί άλλους όρους από το κείμενο· αν ένα απόσπασμα περιγράφει το ίδιο θέμα με διαφορετική διατύπωση, χρησιμοποίησέ το.
3. Μετά από κάθε πρόταση που στηρίζεται σε απόσπασμα, γράψε τον αριθμό του σε αγκύλες, π.χ. [2] ή [2][5]. Μην επινοείς αριθμούς και μη γράφεις μορφές όπως [1(ii)].
4. Αν τα αποσπάσματα καλύπτουν μόνο ένα μέρος της ερώτησης, απάντησε με ό,τι υπάρχει και πες ρητά τι δεν καλύπτεται.
5. ΜΟΝΟ αν κανένα απόσπασμα δεν σχετίζεται με το θέμα της ερώτησης, γράψε ακριβώς: «{REFUSAL}» και σταμάτα.
6. Οι ενδεικτικές δραστηριότητες είναι γενικές για ολόκληρη την υποενότητα και ΔΕΝ αντιστοιχούν σε συγκεκριμένο μαθησιακό αποτέλεσμα. Μην ισχυριστείς ότι μια δραστηριότητα στοχεύει σε συγκεκριμένο αποτέλεσμα, εκτός αν το απόσπασμα το λέει ρητά.
7. Πρότεινε μια δραστηριότητα ως σχετική με την ερώτηση ΜΟΝΟ αν το απόσπασμα περιγράφει ρητά το θέμα της ερώτησης. Αν καμία δραστηριότητα δεν αφορά ρητά το θέμα, πες το καθαρά και παράθεσε μόνο το σχετικό μαθησιακό αποτέλεσμα, αν υπάρχει.
8. Αντέγραψε τους όρους και τα παραδείγματα όπως γράφονται στο απόσπασμα· μην τα αλλάζεις και μη προσθέτεις λεπτομέρειες που δεν υπάρχουν."""

SYSTEM_PLAIN = ("Είσαι βοηθός για νηπιαγωγούς στην Ελλάδα. "
                "Απαντάς στα ελληνικά, σύντομα και πρακτικά.")


# ----------------------------------------------------------------------------
# Καθαρές συναρτήσεις (χωρίς μοντέλο) - εύκολο να δοκιμαστούν
# ----------------------------------------------------------------------------
def short_cite(meta):
    pages = (f"σ. {meta['page_start']}" if meta["page_start"] == meta["page_end"]
             else f"σ. {meta['page_start']}–{meta['page_end']}")
    kind = {"outcome": "αποτελέσματα", "activity": "δραστηριότητες", "intro": "εισαγωγή"}[meta["kind"]]
    return f"{meta['section']} › {meta['subunit_code']} {meta['subunit']} ({kind}), {pages}"


def build_messages(question, hits):
    """Φτιάχνει τα μηνύματα για το μοντέλο. Αν hits=None -> χωρίς ανάκτηση."""
    if hits is None:
        return [{"role": "system", "content": SYSTEM_PLAIN},
                {"role": "user", "content": question}]
    blocks = [f"[{i}] {h['text']}" for i, h in enumerate(hits, 1)]
    user = ("Αποσπάσματα από το Πρόγραμμα Σπουδών:\n\n" + "\n\n".join(blocks)
            + f"\n\nΕρώτηση: {question}")
    return [{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": user}]


def check_citations(answer, n_sources):
    """Επιστρέφει (έγκυροι αριθμοί, άκυροι αριθμοί) που βρέθηκαν στην απάντηση."""
    nums = []
    for group in re.findall(r"\[([^\]]*)\]", answer):
        group = re.sub(r"\([^)]*\)", "", group)       # '[1(ii), 6]' -> '[1, 6]'
        nums += [int(x) for x in re.findall(r"\d+", group)]
    valid = sorted({n for n in nums if 1 <= n <= n_sources})
    invalid = sorted({n for n in nums if not 1 <= n <= n_sources})
    return valid, invalid


def is_refusal(answer):
    """Άρνηση = ολόκληρη η απάντηση είναι η φράση άρνησης (όχι απλώς περιέχει «δεν καλύπτει»
    κάπου μέσα σε μια κανονική απάντηση)."""
    return answer.strip().startswith(REFUSAL[:-1]) and len(answer.strip()) < len(REFUSAL) + 40


# ----------------------------------------------------------------------------
# Το σύστημα
# ----------------------------------------------------------------------------
class RagSystem:
    def __init__(self, llm_path=LLM_PATH, embed_model=MODEL_NAME, index_dir=INDEX_DIR,
                 n_ctx=6144, retrieval="balanced", k=6, k_outcome=3, k_activity=3):
        from llama_cpp import Llama          # εισαγωγή εδώ, ώστε να φορτώνει μόνο όταν χρειάζεται
        from retrieve import Retriever
        self.retriever = Retriever(embed_model, index_dir)
        print(f"Φόρτωση γλωσσικού μοντέλου: {llm_path}")
        # n_gpu_layers=-1: όλα τα επίπεδα στη GPU του M1 (Metal)
        self.llm = Llama(model_path=llm_path, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
        self.retrieval, self.k = retrieval, k
        self.k_outcome, self.k_activity = k_outcome, k_activity

    def answer(self, question, use_rag=True, temperature=0.1, max_tokens=700):
        hits = None
        if use_rag:
            hits = (self.retriever.search_balanced(question, self.k_outcome, self.k_activity)
                    if self.retrieval == "balanced"
                    else self.retriever.search(question, k=self.k))
        messages = build_messages(question, hits)
        out = self.llm.create_chat_completion(messages=messages, temperature=temperature,
                                              max_tokens=max_tokens)
        text = out["choices"][0]["message"]["content"].strip()
        result = {"question": question, "use_rag": use_rag, "answer": text}
        if hits is not None:
            result["prompt"] = messages[1]["content"]
            valid, invalid = check_citations(text, len(hits))
            result.update({
                "refused": is_refusal(text),
                "cited": [{"n": n, "source": short_cite(hits[n - 1]["meta"]),
                           "chunk_id": hits[n - 1]["id"]} for n in valid],
                "invalid_citations": invalid,
                "no_citations": (not valid) and not is_refusal(text),
                "retrieved": [{"n": i, "chunk_id": h["id"], "score": round(h["score"], 3),
                               "source": short_cite(h["meta"])} for i, h in enumerate(hits, 1)],
            })
        return result


def format_result(r, show_prompt=False):
    lines = [f"Ερώτηση: {r['question']}", "", r["answer"], ""]
    if show_prompt and r.get("prompt"):
        lines = [r["prompt"], "", "-" * 60] + lines
    if r["use_rag"]:
        if r["cited"]:
            lines.append("Πηγές που αναφέρονται στην απάντηση:")
            lines += [f"  [{c['n']}] {c['source']}" for c in r["cited"]]
        if r["invalid_citations"]:
            lines.append(f"⚠ Άκυροι αριθμοί παραπομπής (δεν αντιστοιχούν σε απόσπασμα): {r['invalid_citations']}")
        if r["no_citations"]:
            lines.append("⚠ Η απάντηση δεν περιέχει παραπομπές.")
        lines.append("Ανακτήθηκαν: " + ", ".join(
            f"[{x['n']}] {x['chunk_id']} ({x['score']})" for x in r["retrieved"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--batch", help="αρχείο με μία ερώτηση ανά γραμμή")
    ap.add_argument("--out", default="results/answers", help="πρόθεμα αρχείων εξόδου για --batch")
    ap.add_argument("--no-rag", action="store_true", help="χωρίς ανάκτηση (baseline)")
    ap.add_argument("--retrieval", choices=["balanced", "topk"], default="balanced")
    ap.add_argument("-k", type=int, default=6, help="για --retrieval topk")
    ap.add_argument("--k-outcome", type=int, default=3, help="πόσα αποτελέσματα (balanced)")
    ap.add_argument("--k-activity", type=int, default=3, help="πόσες δραστηριότητες (balanced)")
    ap.add_argument("--show-prompt", action="store_true", help="τύπωσε τα αποσπάσματα που πήρε το μοντέλο")
    ap.add_argument("--llm", default=LLM_PATH)
    ap.add_argument("--model", default=MODEL_NAME, help="μοντέλο embeddings")
    ap.add_argument("--index-dir", default=INDEX_DIR)
    args = ap.parse_args()

    if not (args.question or args.interactive or args.batch):
        ap.error("δώσε ερώτηση, ή --interactive, ή --batch αρχείο")

    rag = RagSystem(args.llm, args.model, args.index_dir, retrieval=args.retrieval, k=args.k,
                    k_outcome=args.k_outcome, k_activity=args.k_activity)
    use_rag = not args.no_rag

    if args.interactive:
        print("\nΓράψε ερώτηση (κενή γραμμή για έξοδο).")
        while True:
            q = input("\n> ").strip()
            if not q:
                break
            print("\n" + format_result(rag.answer(q, use_rag), args.show_prompt))
    elif args.batch:
        qs = [ln.strip() for ln in Path(args.batch).read_text(encoding="utf-8").splitlines()
              if ln.strip() and not ln.startswith("#")]
        suffix = "_norag" if args.no_rag else ""
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        results = []
        with open(f"{args.out}{suffix}.md", "w", encoding="utf-8") as md:
            for i, q in enumerate(qs, 1):
                print(f"[{i}/{len(qs)}] {q}")
                r = rag.answer(q, use_rag)
                results.append(r)
                md.write(f"## {i}. {q}\n\n{r['answer']}\n\n")
                if use_rag and r["cited"]:
                    md.write("**Πηγές:**\n" + "\n".join(f"- [{c['n']}] {c['source']}" for c in r["cited"]) + "\n\n")
        with open(f"{args.out}{suffix}.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nΑποθηκεύτηκαν: {args.out}{suffix}.md και {args.out}{suffix}.jsonl")
    else:
        print("\n" + format_result(rag.answer(args.question, use_rag), args.show_prompt))


if __name__ == "__main__":
    main()
