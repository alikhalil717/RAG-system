
import re
import subprocess
from pathlib import Path


PDF = Path(__file__).parents[1] / "data" / "pdfs" / "em8718-e.pdf"


def extract_text_with_pdftotext(pdf_path: Path) -> str:
    cmd = ["pdftotext", str(pdf_path), "-"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {res.stderr}")
    return res.stdout


BOILERPLATE_PATTERNS = (
    r"^archival copy\.",
    r"^for current information, see the osu extension catalog:.*$",
    r"^performance excellence$",
    r"^in the wood products industry$",
    r"^em\s*8718$",
    r"^october 1998$",
    r"^\$2\.50$",
)


def clean_text(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        norm = re.sub(r"\s+", " ", s).lower()
        if any(re.match(pat, norm) for pat in BOILERPLATE_PATTERNS):
            continue
        lines.append(s)
    return "\n".join(lines)


def collapse_ocr_letters(text: str) -> str:
    tokens = text.split()
    out = []
    buf = []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
        else:
            if len(buf) >= 4:
                out.append("".join(buf))
            else:
                out.extend(buf)
            buf = []
            out.append(t)
    if buf:
        if len(buf) >= 4:
            out.append("".join(buf))
        else:
            out.extend(buf)
    return " ".join(out)


def chunk_text(text: str, size: int = 400, overlap: int = 80):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + size)
        chunks.append(text[i:j])
        if j == n:
            break
        i = j - overlap
    return chunks


def lexical_score(chunk: str, query: str) -> int:
    q_terms = re.findall(r"\w+", query.lower())
    text = chunk.lower()
    return sum(text.count(t) for t in q_terms)


def mock_answer(query: str, top_chunk: str, source: str, idx: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", top_chunk.strip())
    ans = sentences[0] if sentences else top_chunk[:200]
    citation = f"[{source} chunk {idx}]"
    return f"{ans} {citation}"


def run_demo():
    raw = extract_text_with_pdftotext(PDF)
    cleaned = clean_text(raw)
    cleaned = collapse_ocr_letters(cleaned)
    chunks = chunk_text(cleaned)

    query = "What is operations research?"
    scored = [(lexical_score(c, query), i, c) for i, c in enumerate(chunks)]
    scored.sort(reverse=True)
    top = [item for item in scored if item[0] > 0][:3]

    print("Query:", query)
    print("Top retrievals:")
    for rank, (score, idx, chunk) in enumerate(top, 1):
        preview = chunk.replace('\n', ' ')[:300]
        print(f"{rank}. em8718-e.pdf (chunk {idx})")
        print("   ", preview, "...\n")

    if top:
        best = top[0]
        answer = mock_answer(query, best[2], "em8718-e.pdf", best[1])
    else:
        answer = "I don't know. [no matching chunks found]"

    print("Assistant answer:")
    print(answer)


if __name__ == '__main__':
    run_demo()
