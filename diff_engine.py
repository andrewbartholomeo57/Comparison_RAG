import torch
import numpy as np
import difflib
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def get_all_documents(vectordb):
    collection = vectordb._collection.get()
    return collection["documents"], collection["metadatas"]


def cosine_similarity_gpu(embeddings_v1, embeddings_v2):
    """Compute the cosine similarity matrix on GPU if available, otherwise on CPU."""
    if torch.cuda.is_available():
        t1 = torch.tensor(embeddings_v1, dtype=torch.float32, device="cuda")
        t2 = torch.tensor(embeddings_v2, dtype=torch.float32, device="cuda")

        t1 = torch.nn.functional.normalize(t1, dim=1)
        t2 = torch.nn.functional.normalize(t2, dim=1)

        sim_matrix = torch.mm(t1, t2.T)
        return sim_matrix.cpu().numpy()
    else:
        return sklearn_cosine_similarity(embeddings_v1, embeddings_v2)


def match_chunks(vectordb_v1, vectordb_v2, embedding_model, threshold=0.6):
    docs_v1, meta_v1 = get_all_documents(vectordb_v1)
    docs_v2, meta_v2 = get_all_documents(vectordb_v2)

    embeddings_v1 = embedding_model.embed_documents(docs_v1)
    embeddings_v2 = embedding_model.embed_documents(docs_v2)

    sim_matrix = cosine_similarity_gpu(embeddings_v1, embeddings_v2)

    used_v2_indices = set()
    seen_pairs = set()
    matches = []

    for i, similarities in enumerate(sim_matrix):
        sorted_indices = np.argsort(similarities)[::-1]

        for idx in sorted_indices:
            if idx in used_v2_indices:
                continue

            score = similarities[idx]

            if score >= threshold:
                pair_key = (docs_v1[i], docs_v2[idx])
                if pair_key in seen_pairs:
                    continue

                seen_pairs.add(pair_key)
                used_v2_indices.add(idx)

                matches.append({
                    "v1_text": docs_v1[i],
                    "v2_text": docs_v2[idx],
                    "similarity": float(score),
                    "v1_metadata": meta_v1[i],
                    "v2_metadata": meta_v2[idx]
                })
                break

    return matches


def detect_change_types(matches, vectordb_v1, vectordb_v2):
    docs_v1, _ = get_all_documents(vectordb_v1)
    docs_v2, _ = get_all_documents(vectordb_v2)

    matched_v1 = set(m["v1_text"] for m in matches)
    matched_v2 = set(m["v2_text"] for m in matches)

    removed = [doc for doc in docs_v1 if doc not in matched_v1]
    added   = [doc for doc in docs_v2 if doc not in matched_v2]

    return {"modified": matches, "removed": removed, "added": added}


def text_diff(text1, text2):
    diff = difflib.unified_diff(
        text1.splitlines(),
        text2.splitlines(),
        lineterm=""
    )
    return "\n".join(diff)


def generate_structural_differences(matches):
    results = []
    for pair in matches:
        diff_result = text_diff(pair["v1_text"], pair["v2_text"])
        results.append({
            "similarity": pair["similarity"],
            "diff": diff_result,
            "v1_metadata": pair["v1_metadata"],
            "v2_metadata": pair["v2_metadata"]
        })
    return results
