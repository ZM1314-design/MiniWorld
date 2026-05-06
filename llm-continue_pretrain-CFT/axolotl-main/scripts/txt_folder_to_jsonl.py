import argparse
import json
import os
from pathlib import Path


def iter_text_chunks(text: str, *, max_chars: int) -> list[str]:
    """
    Split very large files into smaller chunks to avoid giant JSONL rows.
    This is char-based (not token-based) and meant as a safe default.
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            cut = text.rfind("\n\n", start, end)
            if cut == -1:
                cut = text.rfind("\n", start, end)
            if cut != -1 and cut > start + max_chars // 4:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a folder of UTF-8 .txt files into Axolotl pretraining JSONL.")
    ap.add_argument("--txt_dir", required=True, help="Folder containing many .txt files (recursively).")
    ap.add_argument("--out_jsonl", required=True, help="Output .jsonl path.")
    ap.add_argument("--max_chars", type=int, default=200_000, help="Max characters per JSONL row (splits longer files).")
    ap.add_argument("--extensions", default=".txt", help="Comma-separated extensions to include (default: .txt).")
    args = ap.parse_args()

    txt_dir = Path(args.txt_dir)
    out_jsonl = Path(args.out_jsonl)
    exts = tuple(e.strip() for e in args.extensions.split(",") if e.strip())

    if not txt_dir.exists() or not txt_dir.is_dir():
        raise SystemExit(f"txt_dir not found or not a directory: {txt_dir}")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    file_count = 0
    row_count = 0
    byte_count = 0

    with out_jsonl.open("w", encoding="utf-8") as w:
        for p in txt_dir.rglob("*"):
            if not p.is_file():
                continue
            if exts and p.suffix.lower() not in {e.lower() for e in exts}:
                continue

            try:
                raw = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raw = p.read_text(encoding="utf-8", errors="ignore")

            file_count += 1
            byte_count += p.stat().st_size

            for chunk in iter_text_chunks(raw, max_chars=args.max_chars):
                w.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                row_count += 1

    size_mb = byte_count / (1024 * 1024)
    print(f"Done. files={file_count} rows={row_count} input_size≈{size_mb:.1f}MB out={out_jsonl}")


if __name__ == "__main__":
    # For Windows paths, prefer quoting arguments:
    # python scripts/txt_folder_to_jsonl.py --txt_dir "E:\data\agri_txt" --out_jsonl "E:\data\agri.jsonl"
    main()

