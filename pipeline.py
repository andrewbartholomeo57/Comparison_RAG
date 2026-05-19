import uuid
import concurrent.futures
from parser import parse_document
from chunker import chunk_document, convert_chunks_to_documents
from vectordb import create_vector_store, load_embedding_model
from content_classifier import classify_elements, detect_limitations


def process_document(file_path, version_label, session_id):
    """Process a single document through the full RAG pipeline."""
    print(f"[{version_label}] Processing (session: {session_id})")

    elements, _ = parse_document(file_path)

    text_elements, table_elements, image_elements = classify_elements(elements)
    print(f"[{version_label}] Text: {len(text_elements)}, "
          f"Tables: {len(table_elements)}, Images: {len(image_elements)}")

    limitations = detect_limitations(elements)
    print(f"[{version_label}] Limitations: {limitations}")

    chunks = chunk_document(text_elements, table_elements, image_elements)
    print(f"[{version_label}] Chunks: {len(chunks)}")

    docs = convert_chunks_to_documents(chunks, version_label)
    print(f"[{version_label}] Documents: {len(docs)}")

    if not docs:
        raise ValueError(
            f"No documents created for {version_label}. Parsing or chunking failed."
        )

    vectordb = create_vector_store(docs, version_label, session_id)
    return vectordb, limitations


def process_documents_parallel(path_v1, path_v2):
    """Process both documents in parallel using a shared session ID."""
    session_id = uuid.uuid4().hex
    print(f"[Session] New analysis session: {session_id}")

    # Pre-load the singleton model on the main thread before threads start.
    print("[Pipeline] Pre-loading embedding model on main thread...")
    load_embedding_model()
    print("[Pipeline] Embedding model ready. Starting parallel document processing...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_v1 = executor.submit(process_document, path_v1, "v1", session_id)
        future_v2 = executor.submit(process_document, path_v2, "v2", session_id)

        vectordb_v1, limitations_v1 = future_v1.result()
        vectordb_v2, limitations_v2 = future_v2.result()

    return vectordb_v1, limitations_v1, vectordb_v2, limitations_v2
