"""
Project Gutenberg Format Handler
=================================
Handles plain-text (.txt) files downloaded from Project Gutenberg.

Gutenberg files have a specific structure that generic TxtFormatHandler ignores:

  1. A HEADER block before the actual text, containing the PG license preamble,
     title, author, release date, encoding info, etc.  It ends at a line matching:
       *** START OF THE PROJECT GUTENBERG EBOOK <TITLE> ***
     (or the older "*** START OF THIS PROJECT GUTENBERG EBOOK" variant)

  2. The BODY — the actual book content.

  3. A FOOTER block after the body that contains the full PG license text.
     It begins at a line matching:
       *** END OF THE PROJECT GUTENBERG EBOOK <TITLE> ***
     (or the older "*** END OF THIS PROJECT GUTENBERG" variant)

This handler:
  - Strips the header and footer completely.
  - Cleans common Gutenberg artefacts (form-feed chars, excessive blank lines,
    underlining rows of dashes/equals, "[Illustration: ...]" captions).
  - Optionally splits the body into chapters (detected by common heading
    patterns like "CHAPTER I", "Chapter 1", "PART I", "Book I", etc.) and
    yields each chapter as a separate record.
  - Normalises encoding (most Gutenberg files are UTF-8 or Latin-1).
  - Preserves the title/author metadata extracted from the header.

Records yielded always contain:
  {
    'text':    str,          # cleaned text (full body or single chapter)
    'source':  str,          # file path
    'title':   str | None,   # extracted title
    'author':  str | None,   # extracted author
    'chapter': int | None,   # chapter index (1-based) if split_chapters=True
    'chapter_title': str | None,  # chapter heading text
  }
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional, Tuple

from .base import BaseFormatHandler
from xorzen.exceptions import DataFormatError


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Gutenberg start/end markers (case-insensitive)
_START_RE = re.compile(
    r"^\*{3}\s*START\s+OF\s+(?:THIS\s+|THE\s+)?PROJECT\s+GUTENBERG",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"^\*{3}\s*END\s+OF\s+(?:THIS\s+|THE\s+)?PROJECT\s+GUTENBERG",
    re.IGNORECASE,
)

# Metadata lines in the header
_TITLE_RE  = re.compile(r"^Title:\s*(.+)$",  re.IGNORECASE)
_AUTHOR_RE = re.compile(r"^Author:\s*(.+)$", re.IGNORECASE)

# Chapter / part / book headings
_CHAPTER_RE = re.compile(
    r"^\s*"
    r"(?:CHAPTER|Chapter|PART|Part|BOOK|Book|SECTION|Section|CANTO|Canto)"
    r"[\s\.\-]+"
    r"(?:[IVXLCDM]+|\d+|[A-Z][a-z]*)"  # Roman numeral, Arabic, or word
    r"(?:[.\s:—\-].*)?$",               # Optional subtitle
)

# Lines that are pure decoration or artefacts
_DECORATION_RE = re.compile(
    r"^[\s\-=_\*\#\~\^]{5,}$"           # rows of dashes, equals, stars, etc.
)

# Illustration/image captions
_ILLUSTRATION_RE = re.compile(
    r"\[(?:Illustration|Image|Fig\.?|Figure)[^\]]*\]",
    re.IGNORECASE,
)

# Form-feed character (page break in old text files)
_FORM_FEED_RE = re.compile(r"\x0c")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class GutenbergFormatHandler(BaseFormatHandler):
    """
    Format handler for Project Gutenberg plain-text files.

    Example usage::

        >>> from xorzen.data.formats import get_handler
        >>> handler = get_handler('gutenberg')
        >>> for record in handler.read(Path('pg1342.txt')):
        ...     print(record['title'], '|', record['text'][:80])
    """

    FORMAT_NAME    = "gutenberg"
    FILE_EXTENSIONS = [".txt", ".text"]   # shares extension with TxtFormatHandler

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(
        self,
        file_path: Path,
        split_chapters: bool = True,
        min_chapter_tokens: int = 50,
        encoding: str = "utf-8",
        encoding_fallback: str = "latin-1",
        remove_illustrations: bool = True,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """
        Read a Gutenberg text file and yield cleaned records.

        Args:
            file_path:            Path to the .txt file.
            split_chapters:       If True, yield one record per detected chapter.
                                  If False, yield the whole body as one record.
            min_chapter_tokens:   Minimum whitespace-separated tokens a chapter
                                  must have to be yielded (filters out empty
                                  headings that aren't real chapters).
            encoding:             Primary encoding to try (default utf-8).
            encoding_fallback:    Fallback encoding if primary fails.
            remove_illustrations: Strip [Illustration: ...] captions.

        Yields:
            Dict records with keys: text, source, title, author,
            chapter (int|None), chapter_title (str|None).
        """
        raw_lines = self._read_raw(file_path, encoding, encoding_fallback)

        # --- Extract metadata & body ---
        title, author, body_lines = self._parse_structure(raw_lines)

        # --- Clean the body ---
        body_lines = self._clean_lines(body_lines, remove_illustrations)

        source = str(file_path)
        base_record: Dict[str, Any] = {
            "source": source,
            "title":  title,
            "author": author,
        }

        if split_chapters:
            yield from self._yield_chapters(
                body_lines, base_record, min_chapter_tokens
            )
        else:
            text = "\n".join(body_lines).strip()
            if text:
                record = {**base_record, "text": text, "chapter": None, "chapter_title": None}
                self.stats["records_read"] += 1
                yield record

        self.stats["files_processed"] += 1
        self.stats["bytes_processed"] += file_path.stat().st_size

    def write(
        self,
        file_path: Path,
        records: Iterator[Dict[str, Any]],
        **kwargs,
    ):
        """Write records back as plain text (one record per paragraph block)."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for record in records:
                    text = record.get("text", "")
                    if text:
                        f.write(text.strip() + "\n\n")
                        self.stats["records_read"] += 1
            self.stats["files_processed"] += 1
        except Exception as e:
            self.stats["errors"] += 1
            raise DataFormatError(
                format_type="gutenberg",
                reason=f"Failed to write: {e}",
                supported_formats=["txt"],
            )

    def validate(self, file_path: Path) -> bool:
        """Return True if the file contains a Gutenberg start marker."""
        try:
            raw = self._read_raw(file_path, "utf-8", "latin-1")
            for line in raw:
                if _START_RE.match(line):
                    return True
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_raw(
        self, file_path: Path, encoding: str, fallback: str
    ) -> List[str]:
        """Read all lines, trying encoding then fallback."""
        for enc in (encoding, fallback):
            try:
                with open(file_path, "r", encoding=enc, errors="strict") as f:
                    return f.readlines()
            except (UnicodeDecodeError, LookupError):
                continue
        # Last resort: ignore bad bytes
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()

    def _parse_structure(
        self, raw_lines: List[str]
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Locate the START and END markers, extract title/author from header,
        and return the body lines between the markers.
        """
        title:  Optional[str] = None
        author: Optional[str] = None
        body_lines: List[str] = []

        in_header  = True
        in_body    = False

        for raw in raw_lines:
            line = raw.rstrip("\r\n")

            if in_header:
                # Look for metadata
                m = _TITLE_RE.match(line)
                if m:
                    title = m.group(1).strip()

                m = _AUTHOR_RE.match(line)
                if m:
                    author = m.group(1).strip()

                # Detect start of body
                if _START_RE.match(line):
                    in_header = False
                    in_body   = True
                continue

            if in_body:
                # Detect end of body
                if _END_RE.match(line):
                    break
                body_lines.append(line)

        # If no START marker found, treat entire file as body
        # (some Gutenberg mirrors omit the boilerplate)
        if in_header and not in_body:
            body_lines = [l.rstrip("\r\n") for l in raw_lines]

        return title, author, body_lines

    def _clean_lines(
        self, lines: List[str], remove_illustrations: bool
    ) -> List[str]:
        """
        Clean individual lines:
        - Remove form-feed characters
        - Remove decoration rows
        - Optionally strip illustration captions
        - Strip trailing whitespace
        - Collapse runs of more than 2 consecutive blank lines to 2
        """
        cleaned: List[str] = []
        blank_run = 0

        for line in lines:
            # Normalise form feeds to a blank line
            line = _FORM_FEED_RE.sub("", line)
            line = line.rstrip()

            # Drop decoration-only lines
            if _DECORATION_RE.match(line):
                continue

            # Drop illustration captions
            if remove_illustrations and _ILLUSTRATION_RE.search(line):
                line = _ILLUSTRATION_RE.sub("", line).strip()
                if not line:
                    continue

            # Limit consecutive blank lines
            if line == "":
                blank_run += 1
                if blank_run > 2:
                    continue
            else:
                blank_run = 0

            cleaned.append(line)

        return cleaned

    def _yield_chapters(
        self,
        body_lines: List[str],
        base_record: Dict[str, Any],
        min_tokens: int,
    ) -> Iterator[Dict[str, Any]]:
        """
        Split body into chapters by detecting heading lines and yield each
        as a separate record.  If no chapter headings are found, yields the
        entire body as a single record (chapter=1, chapter_title=None).
        """
        # Collect (line_index, heading_text) for each detected chapter heading
        chapter_starts: List[Tuple[int, str]] = []

        for i, line in enumerate(body_lines):
            stripped = line.strip()
            if stripped and _CHAPTER_RE.match(stripped):
                chapter_starts.append((i, stripped))

        if not chapter_starts:
            # No chapters found — emit whole body
            text = "\n".join(body_lines).strip()
            if text and len(text.split()) >= min_tokens:
                self.stats["records_read"] += 1
                yield {
                    **base_record,
                    "text":          text,
                    "chapter":       1,
                    "chapter_title": None,
                }
            return

        # Emit each chapter slice
        boundaries = [idx for idx, _ in chapter_starts]
        boundaries.append(len(body_lines))  # sentinel

        for chapter_idx, (start_line, heading) in enumerate(chapter_starts):
            end_line   = boundaries[chapter_idx + 1]
            # Include the heading in the text
            chapter_lines = body_lines[start_line:end_line]
            text = "\n".join(chapter_lines).strip()

            if not text or len(text.split()) < min_tokens:
                continue  # Skip empty or micro-fragments

            self.stats["records_read"] += 1
            yield {
                **base_record,
                "text":          text,
                "chapter":       chapter_idx + 1,
                "chapter_title": heading,
            }

    def estimate_size(self, file_path: Path) -> int:
        """Rough estimate: count lines that look like chapter headings + 1."""
        count = 1
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if _CHAPTER_RE.match(line.strip()):
                        count += 1
        except Exception:
            pass
        return count


__all__ = ["GutenbergFormatHandler"]
