"""Κοινές ρυθμίσεις για το ευρετήριο (Βήμα 2) και την ανάκτηση."""

# Πολυγλωσσικό μοντέλο embeddings που υποστηρίζει ελληνικά.
# Εναλλακτική (πιο ακριβή αλλά πιο βαριά): "BAAI/bge-m3"
MODEL_NAME = "intfloat/multilingual-e5-base"

INDEX_DIR = "data/index"        # εδώ αποθηκεύεται η βάση διανυσμάτων (ChromaDB)
COLLECTION = "nipiagogeio"
CHUNKS_PATH = "data/processed/chunks.jsonl"


def doc_prefix(model_name):
    """Τα μοντέλα E5 θέλουν πρόθεμα 'passage: ' στα κείμενα και 'query: ' στις ερωτήσεις.
    Αν το παραλείψεις, η ποιότητα της ανάκτησης πέφτει. Το bge-m3 δεν θέλει προθέματα."""
    return "passage: " if "e5" in model_name.lower() else ""


def query_prefix(model_name):
    return "query: " if "e5" in model_name.lower() else ""


def load_model(model_name=MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)
