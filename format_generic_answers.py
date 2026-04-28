"""
format_generic_answers.py
Run this from the project root:  py format_generic_answers.py

Reads prompts/generic_answers.jsonl, repairs any broken/split entries,
and rewrites every entry as clean single-line JSONL.
"""

import json
import re
import sys
from pathlib import Path

JSONL_PATH = Path(__file__).parent / "prompts" / "generic_answers.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_json(text: str):
    """Return parsed dict or None."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def _extract_id_and_answer(raw_block: str):
    """
    Fallback extractor for entries that are structurally valid JSON objects
    but contain unescaped newlines / quotes inside the string values.

    Strategy:
      1. Locate `"id"` value using a simple quoted-string regex.
      2. Grab everything after `"answer":` up to the final closing `}` as
         the raw answer text, then strip surrounding quotes if present.
    """
    # Extract id value
    id_match = re.search(r'"id"\s*:\s*"([^"]+)"', raw_block)
    if not id_match:
        return None

    entry_id = id_match.group(1).strip()

    # Find the start of the answer value (after `"answer":`)
    ans_key_match = re.search(r'"answer"\s*:\s*', raw_block)
    if not ans_key_match:
        return None

    after_key = raw_block[ans_key_match.end():]

    # The answer may or may not be wrapped in quotes at the boundaries.
    # Strip a leading quote if present; strip trailing `"}` / `"  }` variants.
    if after_key.startswith('"'):
        after_key = after_key[1:]

    # Remove trailing closing brace + optional quote
    # Handle patterns like:  "}  or  ."}   or  .""}
    after_key = re.sub(r'["\s]*\}\s*$', '', after_key)

    # Unescape any already-escaped sequences so we store the real text
    answer_text = after_key.replace('\\"', '"').strip()

    if not entry_id or not answer_text:
        return None

    return {"id": entry_id, "answer": answer_text}


def load_entries(path: Path) -> list[dict]:
    """
    Parse the JSONL file robustly:
      - Fast path: each line is valid JSON → great.
      - Slow path: accumulate lines until we get a valid JSON object.
      - Fallback: use regex extraction for structurally broken entries.
    """
    raw = path.read_text(encoding="utf-8-sig")
    lines = raw.splitlines()

    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Fast path
        obj = _try_json(line)
        if obj:
            entries.append(obj)
            i += 1
            continue

        # Slow path — accumulate until valid JSON
        accumulated = line
        found = False
        for j in range(i + 1, len(lines)):
            accumulated += "\n" + lines[j]
            obj = _try_json(accumulated)
            if obj:
                entries.append(obj)
                i = j + 1
                found = True
                break

        if found:
            continue

        # Fallback — regex extraction on the full accumulated block
        # (accumulated already holds everything from line i to end of file
        #  if no valid JSON was found; limit to a reasonable window)
        window = "\n".join(lines[i: i + 60])
        obj = _extract_id_and_answer(window)
        if obj:
            entries.append(obj)
            # Advance past the block: find the line that ends with `}`
            end_idx = i
            for k in range(i, min(i + 60, len(lines))):
                if lines[k].strip().endswith('}') or lines[k].strip().endswith('}"'):
                    end_idx = k
                    break
            i = end_idx + 1
        else:
            # Cannot parse — skip this line and warn
            print(f"  WARNING: could not parse line {i + 1}, skipping: {lines[i][:80]!r}")
            i += 1

    return entries


def write_clean_jsonl(entries: list[dict], path: Path):
    """Write each entry as a clean single-line JSON object."""
    _LITERAL_BSLASH_N = chr(92) + chr(110)  # the two-char sequence  \ n
    lines = []
    for entry in entries:
        # Normalise any literal backslash-n sequences to real newlines so the
        # UI renders them correctly instead of showing '\n' as text.
        ans = entry.get("answer", "")
        if _LITERAL_BSLASH_N in ans:
            entry = dict(entry)  # don't mutate the original
            entry["answer"] = ans.replace(_LITERAL_BSLASH_N, "\n")
        lines.append(json.dumps(entry, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not JSONL_PATH.exists():
        print(f"ERROR: file not found: {JSONL_PATH}")
        sys.exit(1)

    print(f"Reading: {JSONL_PATH}")
    entries = load_entries(JSONL_PATH)
    print(f"  Parsed {len(entries)} entries.")

    # Preview ids in file order
    for idx, e in enumerate(entries, 1):
        print(f"  {idx:>2}. {e.get('id', '(no id)')[:70]}")

    # Write back
    write_clean_jsonl(entries, JSONL_PATH)
    print(f"\nDone. {JSONL_PATH} rewritten with {len(entries)} clean entries.")


if __name__ == "__main__":
    main()
