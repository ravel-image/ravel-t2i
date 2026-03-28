"""
src/kg/scraper.py
──────────────────
Scrapes raw text for a given entity from:
    - Wikipedia        (all domains)
    - Project Gutenberg (literary domain)

Returns combined raw text ready to pass into the extraction prompt.
"""

import re
import logging
import requests
import wikipedia
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Gutenberg search endpoint ─────────────────────────────────────────────────
GUTENBERG_SEARCH_URL = "https://gutendex.com/books"


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def scrape_wikipedia(entity_name: str, wiki_search: str) -> str:
    """
    Fetch filtered Wikipedia page content for an entity.

    Tries wiki_search query first, falls back to entity_name directly.
    Strips boilerplate sections (References, See Also, etc.).

    Args:
        entity_name : human-readable name for logging
        wiki_search : optimized search query from the entity JSON

    Returns:
        Filtered page text, or empty string if not found.
    """
    queries = [wiki_search, entity_name]

    for query in queries:
        try:
            page = wikipedia.page(query, auto_suggest=False)
            text = _filter_wikipedia_content(page.content)
            logger.info(f"  Wikipedia: found '{page.title}' ({len(text)} chars)")
            return text

        except wikipedia.exceptions.DisambiguationError as e:
            # Pick the first option and retry
            logger.warning(f"  Wikipedia: disambiguation for '{query}' — trying '{e.options[0]}'")
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                text = _filter_wikipedia_content(page.content)
                logger.info(f"  Wikipedia: found '{page.title}' via disambiguation")
                return text
            except Exception:
                continue

        except wikipedia.exceptions.PageError:
            logger.warning(f"  Wikipedia: no page found for '{query}'")
            continue

        except Exception as e:
            logger.warning(f"  Wikipedia: error for '{query}': {e}")
            continue

    return ""


def _filter_wikipedia_content(content: str) -> str:
    """
    Strip boilerplate sections from Wikipedia content.
    Keeps the main body, removes references and nav sections.
    """
    sections_to_exclude = [
        "See also", "See Also",
        "References", "Notes",
        "External links", "External Links",
        "Bibliography", "Further reading",
        "In popular culture", "Gallery",
    ]
    lines = content.split("\n")
    filtered = []
    skip = False

    for line in lines:
        # Check if this line is a section header we want to skip
        if any(f"== {s}" in line or f"=={s}" in line for s in sections_to_exclude):
            skip = True
        # New section header that is not excluded — resume
        elif line.startswith("==") and not any(s in line for s in sections_to_exclude):
            skip = False

        if not skip:
            filtered.append(line)

    return "\n".join(filtered).strip()


# ── Project Gutenberg ─────────────────────────────────────────────────────────

def scrape_gutenberg(entity_name: str) -> str:
    """
    Search Project Gutenberg for books mentioning the entity,
    fetch the first matching book's plain text, and extract
    paragraphs containing the entity name.

    Uses the Gutendex API (gutendex.com) — no auth needed.

    Args:
        entity_name : character or entity name to search for

    Returns:
        Extracted relevant paragraphs, or empty string if not found.
    """
    try:
        # Search for books by title or author containing the entity name
        resp = requests.get(
            GUTENBERG_SEARCH_URL,
            params={"search": entity_name},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        if not results:
            logger.warning(f"  Gutenberg: no books found for '{entity_name}'")
            return ""

        # Pick first result that has a plain text download
        text_url = _get_plaintext_url(results)
        if not text_url:
            logger.warning(f"  Gutenberg: no plain text download for '{entity_name}'")
            return ""

        # Fetch the full text
        logger.info(f"  Gutenberg: fetching text from {text_url}")
        text_resp = requests.get(text_url, timeout=30)
        text_resp.raise_for_status()
        full_text = text_resp.text

        # Extract only paragraphs mentioning the entity
        extracted = _extract_relevant_paragraphs(full_text, entity_name)
        logger.info(f"  Gutenberg: extracted {len(extracted)} chars for '{entity_name}'")
        return extracted

    except Exception as e:
        logger.warning(f"  Gutenberg: failed for '{entity_name}': {e}")
        return ""


def _get_plaintext_url(results: list) -> str:
    """
    From Gutendex results, find the plain text download URL.
    Prefers text/plain; falls back to text/html.
    """
    for book in results[:3]:   # check first 3 results
        formats = book.get("formats", {})
        for mime in ["text/plain", "text/plain; charset=utf-8", "text/html"]:
            if mime in formats:
                return formats[mime]
    return ""


def _extract_relevant_paragraphs(full_text: str, entity_name: str, max_chars: int = 4000) -> str:
    """
    Extract paragraphs from Gutenberg text that mention the entity name.
    Caps output at max_chars to avoid token overflow.
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    name_lower = entity_name.lower()

    relevant = [p for p in paragraphs if name_lower in p.lower()]

    if not relevant:
        # Fall back to first 4000 chars of the book
        return full_text[:max_chars]

    combined = "\n\n".join(relevant)
    return combined[:max_chars]


# ── Combined Scraper ──────────────────────────────────────────────────────────

def scrape(entity_name: str, wiki_search: str, domain: str) -> str:
    """
    Main scraping function — combines Wikipedia and Gutenberg
    depending on the domain.

    Args:
        entity_name : entity name
        wiki_search : optimized Wikipedia search query
        domain      : domain string — determines which sources to use

    Returns:
        Combined raw text from all sources, capped at 8000 chars.
    """
    logger.info(f"Scraping: '{entity_name}' (domain={domain})")
    parts = []

    # Wikipedia for all domains
    wiki_text = scrape_wikipedia(entity_name, wiki_search)
    if wiki_text:
        parts.append(f"[Wikipedia]\n{wiki_text}")

    # Project Gutenberg for literary domain
    if domain == "literary":
        gutenberg_text = scrape_gutenberg(entity_name)
        if gutenberg_text:
            parts.append(f"[Project Gutenberg]\n{gutenberg_text}")

    if not parts:
        logger.warning(f"  No source text found for '{entity_name}'")
        return ""

    combined = "\n\n".join(parts)

    # Cap total length to avoid LLM token overflow
    return combined[:8000]
