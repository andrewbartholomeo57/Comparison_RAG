import requests
import re
import concurrent.futures

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# Ollama GPU options: offloads all model layers to GPU when available.
OLLAMA_OPTIONS = {
    "num_gpu": -1,
    "num_thread": 4,
    "num_ctx": 4096,
}


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": OLLAMA_OPTIONS,
        }
    )
    result = response.json()
    return result["response"]


def extract_before_after(diff_text):
    """Convert a unified diff string into clean before/after text."""
    before_lines, after_lines = [], []
    for line in diff_text.splitlines():
        if line.startswith(("@@", "---", "+++")):
            continue
        elif line.startswith("-"):
            before_lines.append(line[1:].strip())
        elif line.startswith("+"):
            after_lines.append(line[1:].strip())
        else:
            text = line.strip()
            before_lines.append(text)
            after_lines.append(text)
    return " ".join(before_lines).strip(), " ".join(after_lines).strip()


def build_clean_diff_payload(diff_results, change_types):
    """Build a numbered list of clean before/after pairs from all change types."""
    entries = []
    idx = 1

    for diff in diff_results:
        raw = diff.get("diff", "")
        if not raw.strip():
            continue
        before, after = extract_before_after(raw)
        if before == after:
            continue
        entries.append((idx, "MODIFIED", before, after))
        idx += 1

    for text in change_types.get("added", []):
        t = text.strip()
        if t:
            entries.append((idx, "ADDED", "Not present", t))
            idx += 1

    for text in change_types.get("removed", []):
        t = text.strip()
        if t:
            entries.append((idx, "REMOVED", t, "Not present"))
            idx += 1

    payload_lines = []
    for i, change_type, doc1, doc2 in entries:
        payload_lines.append(
            f"CHANGE {i} [{change_type}]\n"
            f"  Document 1: {doc1}\n"
            f"  Document 2: {doc2}"
        )

    return entries, "\n\n".join(payload_lines)


def label_changes(clean_payload, total_changes):
    """Ask LLaMA to assign a short meaningful label to every change."""
    prompt = f"""You are a document analyst. Below are {total_changes} numbered changes between two documents.

For each CHANGE, write a SHORT, MEANINGFUL, GOOD LABELS label (3-7 words) describing WHAT aspect changed.
Good labels: "Section 1.2 Title", "Eligibility Criteria", "Dosage Instructions", "Page Range", "Added Footnote", "Removed Warning"
Bad labels: "Change 1", "Text modification", raw sentences copied from the document.

{clean_payload}

OUTPUT FORMAT - one line per change, no other text:
CHANGE 1: <your label>
CHANGE 2: <your label>
...
CHANGE {total_changes}: <your label>
"""
    return ask_ollama(prompt)


def generate_summary(clean_payload, user_prompt, limitation_report):
    """Ask LLaMA to write a concise summary paragraph of the differences."""
    prompt = f"""You are a document comparison assistant.

User Request: {user_prompt}
Interpretation Limitations: {limitation_report}

Here are the differences between the two documents:
{clean_payload}

Write a clear, concise summary paragraph (4-6 sentences) for a non-technical reader explaining:
- The overall nature of the two documents
- The main categories of differences (added sections, wording changes, structural changes, etc.)
- Any content that could not be interpreted (images, charts)

Output ONLY the summary paragraph. No headers, no bullets, no table.
"""
    return ask_ollama(prompt)


def parse_labels(label_response, total_changes):
    """Parse the label response into a dict mapping change index to label string."""
    labels = {}
    for line in label_response.splitlines():
        m = re.match(r'CHANGE\s+(\d+)\s*:\s*(.+)', line.strip(), re.IGNORECASE)
        if m:
            labels[int(m.group(1))] = m.group(2).strip()
    for i in range(1, total_changes + 1):
        if i not in labels:
            labels[i] = f"Change {i}"
    return labels


def explain_differences(diff_results, change_types, user_prompt, limitation_report):
    """Run label and summary LLaMA calls in parallel, then assemble the structured response."""
    entries, clean_payload = build_clean_diff_payload(diff_results, change_types)
    total_changes = len(entries)

    if total_changes == 0:
        return "[SUMMARY]No differences detected between the two documents.[/SUMMARY]\n[TABLE]\n[/TABLE]"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_labels  = executor.submit(label_changes, clean_payload, total_changes)
        future_summary = executor.submit(generate_summary, clean_payload, user_prompt, limitation_report)

        label_response = future_labels.result()
        summary_text   = future_summary.result()

    labels = parse_labels(label_response, total_changes)

    table_lines = []
    for i, change_type, doc1, doc2 in entries:
        aspect = labels.get(i, f"Change {i}")
        table_lines.append(
            f"{aspect} | {_truncate(doc1, 120)} | {_truncate(doc2, 120)}"
        )

    table_str = "\n".join(table_lines)
    return f"[SUMMARY]{summary_text}[/SUMMARY]\n[TABLE]\n{table_str}\n[/TABLE]"


def _truncate(text, max_chars):
    text = " ".join(text.split())
    return text[:max_chars].rstrip() + "..." if len(text) > max_chars else text


def parse_llm_response(response_text):
    """Extract summary and table rows from the structured LLM response."""
    summary = ""
    table_rows = []

    summary_match = re.search(r'\[SUMMARY\](.*?)\[/SUMMARY\]', response_text, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary = summary_match.group(1).strip()

    table_match = re.search(r'\[TABLE\](.*?)\[/TABLE\]', response_text, re.DOTALL | re.IGNORECASE)
    if table_match:
        table_rows = _parse_pipe_table(table_match.group(1).strip())

    if not summary:
        summary = response_text

    return summary, table_rows


def _parse_pipe_table(text):
    rows, seen = [], set()
    for line in text.splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        if re.match(r'^[\|\-\s:]+$', line):
            continue
        cols = [c.strip() for c in line.split("|")]
        cols = [c for c in cols if c != ""]
        if len(cols) < 3:
            continue
        aspect, doc1, doc2 = cols[0], cols[1], cols[2]
        if aspect.lower() in ("aspect", "#", "category", "item", "difference"):
            continue
        key = (aspect.lower(), doc1.lower(), doc2.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append({"Aspect": aspect, "Document 1": doc1, "Document 2": doc2})
    return rows
