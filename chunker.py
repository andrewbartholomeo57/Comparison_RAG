from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document


def chunk_document(text_elements, table_elements, image_elements):

    all_elements = []

    if text_elements:
        all_elements.extend(text_elements)

    if table_elements:
        all_elements.extend(table_elements)

    if image_elements:
        all_elements.extend(image_elements)

    if not all_elements:
        return []

    chunks = chunk_by_title(
        all_elements,
        max_characters=2000,
        new_after_n_chars=1800,
        combine_text_under_n_chars=500
    )

    return chunks


def convert_chunks_to_documents(chunks, version):

    documents = []

    if not chunks:
        return documents

    for chunk in chunks:

        metadata = {}

        if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "to_dict"):
            metadata = chunk.metadata.to_dict()

        limitations = []

        if hasattr(chunk.metadata, "orig_elements"):

            for el in chunk.metadata.orig_elements:

                if hasattr(el, "category"):

                    if el.category == "Image":
                        limitations.append("Contains Image")

                    if el.category == "Table":
                        limitations.append("Contains Table")

        metadata["limitations"] = limitations if limitations else ["None"]
        metadata["version"] = version

        documents.append(
            Document(
                page_content=str(chunk),
                metadata=metadata
            )
        )

    return documents
