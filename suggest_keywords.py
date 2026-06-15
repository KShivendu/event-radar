#!/usr/bin/env python3
"""
Suggest Luma search keywords from existing events DB.

Algorithm:
1. Extract noun phrases from event names + descriptions
2. Score by TF-IDF (distinctive to this corpus, not just frequent)
3. Deduplicate overlapping phrases (keep longest)
4. Filter out phrases already covered by current keywords
5. Print ranked candidates for review
"""
import json
import re
import sqlite3
from collections import Counter
from math import log
from pathlib import Path

DB = Path(__file__).parent / "events.db"
KEYWORDS_FILE = Path(__file__).parent / "search_keywords.json"

# Generic stop-phrases to ignore even if frequent
STOPLIST = {
    "ai", "san francisco", "sf", "event", "events", "meetup", "happy hour",
    "workshop", "summit", "conference", "community", "networking", "panel",
    "talk", "talks", "speaker", "speakers", "join", "team", "company",
    "new york", "nyc", "bay area", "tech", "startup", "startups",
    "use case", "use cases", "best practice", "best practices",
    "deep dive", "fireside chat", "q&a", "office hour", "office hours",
    "come", "learn", "build", "world", "future", "next", "way",
}


def load_texts() -> list[str]:
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT name, description FROM events WHERE start_datetime >= datetime('now')"
    ).fetchall()
    conn.close()
    texts = []
    for name, desc in rows:
        texts.append((name or "").strip())
        if desc:
            # First sentence of description only — rest is usually boilerplate
            texts.append(re.split(r'[.!?\n]', desc)[0].strip())
    return [t for t in texts if t]


def extract_noun_phrases(texts: list[str]) -> list[list[str]]:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    doc_phrases = []
    for doc in nlp.pipe(texts, batch_size=64):
        phrases = []
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower().strip()
            phrase = re.sub(r'\s+', ' ', phrase)
            # Remove leading determiners/articles
            phrase = re.sub(r'^(the|a|an|our|your|this|that|these|those|its|their|with|for|on|in|at|of)\s+', '', phrase)
            phrase = phrase.strip()
            if len(phrase) >= 4 and phrase not in STOPLIST:
                phrases.append(phrase)
        doc_phrases.append(phrases)
    return doc_phrases


def tfidf_score(doc_phrases: list[list[str]]) -> dict[str, float]:
    N = len(doc_phrases)
    tf: Counter = Counter()
    df: Counter = Counter()
    for phrases in doc_phrases:
        tf.update(phrases)
        df.update(set(phrases))

    scores = {}
    for phrase, freq in tf.items():
        if df[phrase] < 2:  # must appear in at least 2 docs
            continue
        idf = log(N / df[phrase])
        scores[phrase] = freq * idf
    return scores


def dedup_overlapping(phrases: list[str]) -> list[str]:
    """Keep longer phrase if it contains a shorter one (e.g. keep 'agent memory' over 'memory')."""
    kept = []
    sorted_phrases = sorted(phrases, key=len, reverse=True)
    for phrase in sorted_phrases:
        if not any(phrase in longer for longer in kept):
            kept.append(phrase)
    return kept


def main():
    current_keywords = {k.lower() for k in json.loads(KEYWORDS_FILE.read_text())}

    print("Loading events...", end=" ", flush=True)
    texts = load_texts()
    print(f"{len(texts)} texts")

    print("Extracting noun phrases...", end=" ", flush=True)
    doc_phrases = extract_noun_phrases(texts)
    print("done")

    scores = tfidf_score(doc_phrases)

    candidates = {}
    for p, s in scores.items():
        words = p.split()
        if p in STOPLIST: continue
        if p in current_keywords: continue
        if len(words) > 4: continue               # too long to be a useful search term
        if len(words) == 1 and len(p) < 5: continue  # too short/generic
        if re.search(r'[|/\\•·@#\d]', p): continue   # symbols or numbers = event-specific
        if re.search(r'\b(party|dinner|run|coffee|comedy|pickleball|trivia|hackathon|lounge|warming|afterparty)\b', p): continue
        if any(stop in words for stop in {"event", "meetup", "workshop", "talk", "night", "day", "hour"}): continue
        candidates[p] = s

    ranked = sorted(candidates, key=lambda p: candidates[p], reverse=True)
    ranked = dedup_overlapping(ranked)

    print(f"\nTop keyword suggestions (not in current list):\n")
    print(f"  {'PHRASE':<35} SCORE   IN_CURRENT")
    print(f"  {'-'*35} ------  ----------")
    for phrase in ranked[:40]:
        in_current = "✓" if any(phrase in k or k in phrase for k in current_keywords) else ""
        print(f"  {phrase:<35} {candidates[phrase]:6.1f}  {in_current}")

    print(f"\nCurrent keywords ({len(current_keywords)}):")
    for k in sorted(current_keywords):
        print(f"  - {k}")


if __name__ == "__main__":
    main()
