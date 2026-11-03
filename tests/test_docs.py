"""The guide (docs/PROJECT_GUIDE.md) explains the brief (docs/PROJECT_BRIEF.md): chapter n covers section Sn.

Each guide chapter records, under its title, the hash of the brief section it was last reviewed against
(`Synced: Sn <hash>`). When a brief section changes, `test_guide_chapters_match_brief_sections` names the chapters to
review. After reviewing chapter n against section Sn, copy the line the failure prints into the chapter;
`python tests/test_docs.py` prints the current line for every section.
"""
import hashlib
import re
import unicodedata
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
BRIEF = DOCS / "PROJECT_BRIEF.md"
GUIDE = DOCS / "PROJECT_GUIDE.md"
ASCII_DOCS = (BRIEF, GUIDE, DOCS / "README.md", DOCS / "literature" / "index.md")

_SECTION = re.compile(r"^## S(\d+)\. ")
_CHAPTER = re.compile(r"^## Chapter (\d+)\. ")
_SYNCED = re.compile(r"^Synced: S(\d+) (\S+)$")


def _blocks(text: str, heading: re.Pattern) -> dict[int, list[str]]:
    """Lines under each numbered heading, up to the next level-2 heading."""
    blocks: dict[int, list[str]] = {}
    current = None
    for line in text.splitlines():
        match = heading.match(line)
        if match:
            current = int(match.group(1))
            blocks[current] = []
        elif line.startswith("## "):
            current = None
        elif current is not None:
            blocks[current].append(line.rstrip())
    return blocks


def section_hashes(brief_text: str) -> dict[int, str]:
    return {n: hashlib.sha256("\n".join(lines).strip().encode("utf-8")).hexdigest()[:8]
            for n, lines in _blocks(brief_text, _SECTION).items()}


def synced_lines(guide_text: str) -> dict[int, tuple[int, str] | None]:
    """Chapter number -> (section number, hash) from its `Synced:` line, or None if the line is missing."""
    out: dict[int, tuple[int, str] | None] = {}
    for n, lines in _blocks(guide_text, _CHAPTER).items():
        found = [_SYNCED.match(line) for line in lines if _SYNCED.match(line)]
        out[n] = (int(found[0].group(1)), found[0].group(2)) if len(found) == 1 else None
    return out


def test_guide_chapters_match_brief_sections():
    current = section_hashes(BRIEF.read_text(encoding="utf-8"))
    recorded = synced_lines(GUIDE.read_text(encoding="utf-8"))
    problems = []
    for n, digest in sorted(current.items()):
        if n not in recorded:
            problems.append(f"brief S{n} has no guide chapter {n}")
        elif recorded[n] is None:
            problems.append(f"guide chapter {n} has no single 'Synced:' line; add 'Synced: S{n} {digest}'")
        elif recorded[n] != (n, digest):
            problems.append(f"guide chapter {n} was reviewed against an older S{n}: review it, then set "
                            f"'Synced: S{n} {digest}'")
    problems += [f"guide chapter {n} has no brief section S{n}" for n in sorted(set(recorded) - set(current))]
    assert not problems, "\n".join(problems)


def test_docs_use_ascii_notation():
    """Notation is ASCII; only Latin letters with diacritics (author names) are allowed beyond it."""
    problems = []
    for path in ASCII_DOCS:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            bad = sorted({ch for ch in line if ord(ch) > 127 and "LATIN" not in unicodedata.name(ch, "")})
            if bad:
                problems.append(f"{path.relative_to(DOCS)}:{number}: {''.join(bad)}")
    assert not problems, "\n".join(problems)


if __name__ == "__main__":
    for n, digest in sorted(section_hashes(BRIEF.read_text(encoding="utf-8")).items()):
        print(f"Synced: S{n} {digest}")
