import threading
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Singleton embedding model with a thread lock to prevent parallel-init crashes.
_embedding_model = None
_model_lock = threading.Lock()


def get_device():
    """Detect best available device: CUDA GPU > MPS (Apple) > CPU."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[GPU] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[GPU] Using Apple MPS")
    else:
        device = "cpu"
        print("[CPU] No GPU detected, falling back to CPU")
    return device


def load_embedding_model():
    """Return the singleton embedding model, initialising it on first call (thread-safe)."""
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    with _model_lock:
        if _embedding_model is None:
            device = get_device()
            print(f"[Embedding] Initialising model on device={device} ...")
            _embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 64,
                },
            )
            print("[Embedding] Model ready.")

    return _embedding_model


def create_vector_store(documents, version_label, session_id):
    """Create a fresh in-memory Chroma vector store isolated to this session."""
    embedding = load_embedding_model()

    collection_name = f"rag_{session_id}_{version_label}"

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name=collection_name,
    )

    print(f"[{version_label}] Collection '{collection_name}' created in memory.")
    return vectordb
