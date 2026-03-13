"""
Robust output extraction for Qwen3 (and other models with <think> tags).

Handles these known output patterns:
  1. <think>...</think>\nAnswer        — standard thinking mode
  2. Answer\n<think>...</think>          — answer first, thinking after
  3. Garbage\nAnswer\nThinking...        — prefix junk + answer + rambling
  4. **Answer**: Yes                     — markdown-formatted answer
  5. Plain answer with no tags           — clean output
"""

import re
from typing import Optional


def _strip_thinking(raw: str) -> str:
    """Remove all <think>...</think> blocks (greedy, handles nested/malformed)."""
    # Remove complete <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # If there's an unclosed <think> at the end, remove from <think> onward
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


# Patterns that indicate the start of unstructured "thinking out loud"
# even when <think> tags are absent (common with Qwen3 no-think mode leaks)
_THINKING_PREFIXES = re.compile(
    r"^(okay|ok|let me|let's|hmm|so|well|alright|now|first)[,\s]",
    re.IGNORECASE,
)


def _is_thinking_line(line: str) -> bool:
    """Heuristic: does this line look like internal reasoning, not an answer?"""
    stripped = line.strip()
    if _THINKING_PREFIXES.match(stripped):
        return True
    # Very long lines are almost certainly reasoning, not a short answer
    if len(stripped) > 200:
        return True
    return False


def _get_post_think(raw: str) -> Optional[str]:
    """Get content after the last </think> tag, if present."""
    if "</think>" in raw:
        after = raw.split("</think>")[-1].strip()
        return after if after else None
    return None


def extract_yes_no(raw: str) -> bool:
    """
    Extract a Yes/No answer from model output.
    Designed for binary decision steps (detection, judge verification).

    Strategy (in priority order):
      1. If </think> present → look at post-think content only
      2. Look for explicit "Answer: Yes/No" pattern anywhere
      3. Check first non-empty line after stripping think blocks
      4. Scan all lines for a standalone Yes/No
      5. Count occurrences as fallback
    """
    # Strategy 1: post-think content
    post = _get_post_think(raw)
    if post is not None:
        # Check first substantive line after </think>
        for line in post.split("\n"):
            line_clean = line.strip().strip("*").strip(".").strip()
            if line_clean.lower() in ("yes", "no"):
                return line_clean.lower() == "yes"
        # Check if post-think contains yes/no anywhere
        post_lower = post.lower()
        if "yes" in post_lower and "no" not in post_lower:
            return True
        if "no" in post_lower and "yes" not in post_lower:
            return False

    # Strategy 2: explicit "Answer:" pattern
    m = re.search(
        r"\*?\*?[Aa]nswer\*?\*?\s*[:：]\s*(yes|no)",
        raw, re.IGNORECASE,
    )
    if m:
        return m.group(1).lower() == "yes"

    # Strategy 3: strip thinking, check first non-empty non-thinking line
    cleaned = _strip_thinking(raw)
    if cleaned:
        for line in cleaned.split("\n"):
            line_clean = line.strip().strip("*").strip(".").strip()
            if line_clean.lower() in ("yes", "no"):
                return line_clean.lower() == "yes"
            # Skip lines that look like reasoning leaks
            if _is_thinking_line(line):
                continue
            # If we hit a short non-thinking line, check for yes/no within it
            if "yes" in line_clean.lower() or "no" in line_clean.lower():
                if "yes" in line_clean.lower():
                    return True
                return False

    # Strategy 4: scan ALL lines (outside think blocks) for standalone yes/no
    for line in cleaned.split("\n"):
        line_lower = line.strip().lower()
        # Match lines that are just "yes"/"no" possibly with punctuation/markdown
        if re.match(r"^[\s*]*yes[\s.*!]*$", line_lower):
            return True
        if re.match(r"^[\s*]*no[\s.*!]*$", line_lower):
            return False

    # Strategy 5: fallback — count occurrences in cleaned text
    cleaned_lower = cleaned.lower()
    yes_count = len(re.findall(r"\byes\b", cleaned_lower))
    no_count = len(re.findall(r"\bno\b", cleaned_lower))
    if yes_count > 0 or no_count > 0:
        return yes_count > no_count

    # Ultimate fallback: assume no
    return False


def extract_text_answer(raw: str) -> str:
    """
    Extract a free-text answer (e.g., identification of secret type).
    Used for steps that expect descriptive text, not yes/no.

    Strategy:
      1. If </think> present → return post-think content
      2. Look for "Answer:" pattern
      3. Strip thinking blocks, return first substantial line
    """
    # Strategy 1: post-think content
    post = _get_post_think(raw)
    if post is not None:
        # Return first non-trivial line (skip very short junk)
        lines = [l.strip() for l in post.split("\n") if l.strip()]
        if lines:
            # If first line looks like a label ("Answer:"), extract the value
            m = re.match(r"\*?\*?[Aa]nswer\*?\*?\s*[:：]\s*(.*)", lines[0])
            if m and m.group(1).strip():
                return m.group(1).strip().strip("*").strip()
            return lines[0].strip("*").strip()
        return post

    # Strategy 2: explicit "Answer:" anywhere in text
    m = re.search(
        r"\*?\*?[Aa]nswer\*?\*?\s*[:：]\s*(.+?)(?:\n|$)",
        raw, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().strip("*").strip()

    # Strategy 3: strip thinking, return first substantial non-thinking line
    cleaned = _strip_thinking(raw)
    for line in cleaned.split("\n"):
        stripped = line.strip()
        # Skip empty, separator, or meta lines
        if not stripped or stripped.startswith("=") or stripped.startswith("---"):
            continue
        # Skip lines that look like reasoning leaks
        if _is_thinking_line(stripped):
            continue
        return stripped.strip("*").strip()

    # If everything looked like thinking, return the last short-ish line
    for line in reversed(cleaned.split("\n")):
        stripped = line.strip()
        if stripped and len(stripped) < 200:
            return stripped.strip("*").strip()

    return cleaned[:200] if cleaned else raw[:200]


def parse_evidence(raw: str) -> list[str]:
    """
    Extract EVIDENCE lines from grounding output.
    Looks in BOTH thinking blocks and post-think content,
    since models sometimes put structured output inside <think>.

    Handles:
      - EVIDENCE: "quoted text"
      - EVIDENCE: quoted text (no quotes)
      - - "quoted text" (bullet list of evidence)
      - Numbered lists: 1. "evidence" or 1) "evidence"
    """
    evidence = []
    seen = set()

    # Search everywhere — both inside and outside think blocks
    full_text = raw

    # Pattern 1: EVIDENCE: lines
    for m in re.finditer(
        r'EVIDENCE\s*[:：]\s*["""]?(.+?)["""]?\s*$',
        full_text, re.MULTILINE | re.IGNORECASE,
    ):
        val = m.group(1).strip().strip('"').strip('"').strip('"').strip()
        if val and val not in seen:
            evidence.append(val)
            seen.add(val)

    if evidence:
        return evidence

    # Pattern 2: bullet list items with quotes (after a header like "Evidence:")
    for m in re.finditer(
        r'[-•]\s*["""](.+?)["""]',
        full_text, re.MULTILINE,
    ):
        val = m.group(1).strip()
        if val and val not in seen:
            evidence.append(val)
            seen.add(val)

    if evidence:
        return evidence

    # Pattern 3: numbered list items
    for m in re.finditer(
        r'\d+[.)]\s*["""]?(.+?)["""]?\s*$',
        full_text, re.MULTILINE,
    ):
        val = m.group(1).strip().strip('"').strip('"').strip('"').strip()
        # Filter out lines that are too long (likely explanations, not evidence quotes)
        if val and val not in seen and len(val) < 500:
            evidence.append(val)
            seen.add(val)

    return evidence