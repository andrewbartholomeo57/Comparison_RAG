import torch
from unstructured.partition.auto import partition


def get_device_strategy():
    """Return 'hi_res' when CUDA is available, otherwise 'fast'."""
    if torch.cuda.is_available():
        print("[GPU] Parser using hi_res strategy with CUDA")
        return "hi_res"
    else:
        print("[CPU] Parser using fast strategy (no GPU detected)")
        return "fast"


def parse_document(file_path):
    """Parse a document into structured elements and return element counts."""
    strategy = get_device_strategy()

    kwargs = {
        "filename": file_path,
        "strategy": strategy,
    }

    if strategy == "hi_res" and torch.cuda.is_available():
        kwargs["hi_res_model_name"] = "yolox"

    elements = partition(**kwargs)

    text_count = 0
    table_count = 0
    image_count = 0

    for el in elements:
        if not hasattr(el, "category"):
            continue
        if el.category in ["NarrativeText", "Title", "ListItem"]:
            text_count += 1
        elif el.category == "Table":
            table_count += 1
        elif el.category == "Image":
            image_count += 1

    metadata = {
        "text_count": text_count,
        "table_count": table_count,
        "image_count": image_count
    }

    return elements, metadata
