def flatten_elements(elements):
    flat = []
    for el in elements:
        if isinstance(el, list):
            flat.extend(el)
        else:
            flat.append(el)
    return flat


def classify_elements(elements):

    elements = flatten_elements(elements)

    text_elements = []
    table_elements = []
    image_elements = []

    for el in elements:

        if not hasattr(el, "category"):
            continue

        if el.category == "Table":
            table_elements.append(el)

        elif el.category == "Image":
            image_elements.append(el)

        else:
            text_elements.append(el)

    return text_elements, table_elements, image_elements


def detect_limitations(elements):

    elements = flatten_elements(elements)

    has_image = False
    has_table = False

    for el in elements:

        if not hasattr(el, "category"):
            continue

        if el.category == "Image":
            has_image = True

        if el.category == "Table":
            has_table = True

    limitations = []

    if has_image:
        limitations.append("Document contains images")

    if has_table:
        limitations.append("Document contains tables")

    if not limitations:
        limitations.append("No structural limitations detected")

    return limitations
