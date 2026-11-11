"""
Βήμα 2α - Δημιουργία ευρετηρίου: embeddings για κάθε chunk και αποθήκευση σε ChromaDB.

Χρήση:
  python src/embed_index.py
  python src/embed_index.py --model BAAI/bge-m3      # άλλο μοντέλο embeddings

Την πρώτη φορά το μοντέλο κατεβαίνει αυτόματα από το Hugging Face (~1 GB,
χρειάζεται internet) και μετά μένει στη cache του υπολογιστή.
"""
import argparse
import json
from pathlib import Path

import chromadb

from common import (CHUNKS_PATH, COLLECTION, INDEX_DIR, MODEL_NAME,
                    doc_prefix, load_model)


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=CHUNKS_PATH)
    ap.add_argument("--index-dir", default=INDEX_DIR)
    ap.add_argument("--model", default=MODEL_NAME)
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    print(f"Φορτώθηκαν {len(chunks)} chunks από {args.chunks}")

    print(f"Φόρτωση μοντέλου embeddings: {args.model}")
    model = load_model(args.model)

    # Το κείμενο που κάνουμε embedding περιέχει και την επικεφαλίδα (ιεραρχία).
    # Αν υπάρχει 'embed_text' (βλ. ingest.py --embed-mode), γίνεται embedding αυτό·
    # αλλιώς η επικεφαλίδα μαζί με το σώμα.
    texts = [doc_prefix(args.model) + c.get("embed_text", c["text"]) for c in chunks]
    # normalize_embeddings=True -> μοναδιαία διανύσματα, ώστε η cosine similarity
    # να ισούται με εσωτερικό γινόμενο.
    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=16)

    Path(args.index_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=args.index_dir)
    # Ξαναχτίζουμε από την αρχή, ώστε το ευρετήριο να ταιριάζει πάντα με τα chunks.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    col.add(
        ids=[c["id"] for c in chunks],
        embeddings=[e.tolist() for e in embeddings],
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "section": c["section"],
            "subunit_code": c["subunit_code"],
            "subunit": c["subunit"],
            "kind": c["kind"],
            "category": c["category"] or "",     # η Chroma δεν δέχεται None
            "topics": " | ".join(c["topics"]),   # ούτε λίστες
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "citation": c["citation"],
        } for c in chunks],
    )
    print(f"Αποθηκεύτηκαν {col.count()} διανύσματα στο {args.index_dir} "
          f"(διάσταση {len(embeddings[0])})")


if __name__ == "__main__":
    main()
