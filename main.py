from pipeline import process_document
from vectordb import load_embedding_model
from diff_engine import match_chunks, generate_structural_differences
from diff_engine import detect_change_types
from ollama_engine import explain_differences

import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_v1 = os.path.join(BASE_DIR, "data", "doc_v1.pdf")
file_v2 = os.path.join(BASE_DIR, "data", "doc_v2.pdf")


def main():

    # Step 1 - Process documents
    vectordb_v1, limitations_v1 = process_document(file_v1, "v1")
    vectordb_v2, limitations_v2 = process_document(file_v2, "v2")

    # Step 2 - Load embedding model
    embedding_model = load_embedding_model()

    # Step 3 - Match chunks
    matches = match_chunks(
        vectordb_v1,
        vectordb_v2,
        embedding_model
    )

    # Step 4 - Generate structural diff
    diff_results = generate_structural_differences(matches)

    # Step 5 - Detect change types
    change_types = detect_change_types(
        matches,
        vectordb_v1,
        vectordb_v2
    )

    print("Added:", len(change_types["added"]))
    print("Removed:", len(change_types["removed"]))
    print("Modified:", len(change_types["modified"]))

    # Step 6 - Combine limitation reports
    limitation_report = {
        "v1": limitations_v1,
        "v2": limitations_v2
    }

    print("\nInterpretation Limitations:")
    print(limitation_report)

    # Step 7 - User prompt
    user_prompt = input("\nEnter analysis prompt: ")

    # Step 8 - OLLAMA explanation
    explanation = explain_differences(
        diff_results,
        user_prompt,
        limitation_report
    )

    print("\n===== OLLAMA SUMMARY =====")
    print(explanation)

    # Step 9 - Print sample structural diffs
    for diff in diff_results[:3]:
        print("Similarity:", diff["similarity"])
        print(diff["diff"])
        print("=" * 50)

    print("V1 count:", vectordb_v1._collection.count())
    print("V2 count:", vectordb_v2._collection.count())


if __name__ == "__main__":
    main()
