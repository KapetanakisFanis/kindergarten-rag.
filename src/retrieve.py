"""
Βήμα 2β - Ανάκτηση: για μια ερώτηση επιστρέφει τα πιο σχετικά αποσπάσματα.

Χρήση από το Terminal (δοκιμή χωρίς γλωσσικό μοντέλο):
  python src/retrieve.py "Πώς δουλεύω τη φωνολογική επίγνωση;"
  python src/retrieve.py "ερώτηση..." -k 8
  python src/retrieve.py "ερώτηση..." --section Μαθηματικά
  python src/retrieve.py "ερώτηση..." --kind activity       # μόνο δραστηριότητες
  python src/retrieve.py "ερώτηση..." --balanced            # 3 αποτελέσματα + 3 δραστηριότητες + 1 εισαγωγή
  python src/retrieve.py "ερώτηση..." --model BAAI/bge-m3 --index-dir data/index_bge

Χρήση από κώδικα (θα τη χρησιμοποιήσουμε στο Βήμα 3):
  from retrieve import Retriever
  hits = Retriever().search_balanced("ερώτηση")
"""
import argparse

import chromadb

from common import COLLECTION, INDEX_DIR, MODEL_NAME, load_model, query_prefix


class Retriever:
    def __init__(self, model_name=MODEL_NAME, index_dir=INDEX_DIR):
        self.model_name = model_name
        self.model = load_model(model_name)
        client = chromadb.PersistentClient(path=index_dir)
        self.col = client.get_collection(COLLECTION)

    def _embed(self, query):
        return self.model.encode([query_prefix(self.model_name) + query],
                                 normalize_embeddings=True)[0].tolist()

    def _query(self, qvec, k, section=None, kind=None):
        conds = []
        if section:
            conds.append({"section": section})
        if kind:
            conds.append({"kind": kind})
        where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})
        res = self.col.query(query_embeddings=[qvec], n_results=k, where=where)
        hits = []
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "score": 1.0 - res["distances"][0][i],   # cosine similarity
                "text": res["documents"][0][i],
                "citation": res["metadatas"][0][i]["citation"],
                "meta": res["metadatas"][0][i],
            })
        return hits

    def search(self, query, k=5, section=None, kind=None):
        """Απλή αναζήτηση: τα k πιο κοντινά chunks."""
        return self._query(self._embed(query), k, section, kind)

    def search_balanced(self, query, k_outcome=3, k_activity=3, k_intro=1, section=None):
        """Ισορροπημένη αναζήτηση: ξεχωριστά τα καλύτερα 'αποτελέσματα', 'δραστηριότητες'
        και 'εισαγωγικά'. Έτσι ένας τύπος chunk δεν καταλαμβάνει όλες τις θέσεις.
        Το εισαγωγικό κείμενο επιλέγεται ΜΟΝΟ από την ενότητα (Γλώσσα/Μαθηματικά) του
        καλύτερου αποτελέσματος, ώστε να μη μπαίνει άσχετη εισαγωγή άλλης ενότητας.
        Το αποτέλεσμα ταξινομείται κατά score."""
        qvec = self._embed(query)
        hits = []
        for kind, k in (("outcome", k_outcome), ("activity", k_activity)):
            if k > 0:
                hits += self._query(qvec, k, section, kind)
        if k_intro > 0 and hits:
            best_section = max(hits, key=lambda h: h["score"])["meta"]["section"]
            hits += self._query(qvec, k_intro, section or best_section, "intro")
        return sorted(hits, key=lambda h: h["score"], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--section", choices=["Γλώσσα", "Μαθηματικά"], default=None)
    ap.add_argument("--kind", choices=["intro", "outcome", "activity"], default=None)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--k-outcome", type=int, default=3)
    ap.add_argument("--k-activity", type=int, default=3)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--index-dir", default=INDEX_DIR)
    args = ap.parse_args()

    r = Retriever(args.model, args.index_dir)
    if args.balanced:
        hits = r.search_balanced(args.query, args.k_outcome, args.k_activity, 1, args.section)
    else:
        hits = r.search(args.query, k=args.k, section=args.section, kind=args.kind)
    print(f"\nΕρώτηση: {args.query}\n")
    for rank, h in enumerate(hits, 1):
        m = h["meta"]
        pages = (f"σ. {m['page_start']}" if m["page_start"] == m["page_end"]
                 else f"σ. {m['page_start']}–{m['page_end']}")
        print(f"[{rank}] score={h['score']:.3f}  {h['id']}  {m['section']} › "
              f"{m['subunit_code']} {m['subunit']}  ({m['kind']}, {pages})")
        body = h["text"].split("\n", 1)[1] if "\n" in h["text"] else h["text"]
        print("    " + body[:220].replace("\n", " ") + ("…" if len(body) > 220 else ""))
        print()


if __name__ == "__main__":
    main()
