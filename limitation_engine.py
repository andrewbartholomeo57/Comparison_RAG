def detect_interpretation_limitations(meta_v1, meta_v2):

    report = []

    if meta_v1["table_count"] > 0 or meta_v2["table_count"] > 0:
        report.append("Tables detected but not fully interpreted.")

    if meta_v1["image_count"] > 0 or meta_v2["image_count"] > 0:
        report.append("Images or charts detected but not interpreted.")

    if not report:
        report.append("All detected content types were text and interpreted successfully.")

    return "\n".join(report)
