"""One entry point for the full per-event people flow, shared by the CLI
(find_people.py) and the web UI (ui.py):

    collect (hosts + featured guests)  →  enrich contacts + faces  →  rank

Each stage is optional and persisted to events.db as it runs.
"""
import enrich
import enrich_contacts
from scraper.db import save_ranking
from scraper.sources import luma_people


def find_people(url_or_id: str, *, contacts: bool = True, contacts_web: bool = False,
                rank: bool = True, rank_web: bool = False) -> tuple[dict, list[dict]]:
    """Run the pipeline for one event. Returns (summary, people) where each person
    dict carries collected + (optionally) enriched + (optionally) ranked fields."""
    summary = luma_people.collect(url_or_id)   # persists people, returns summary + people
    eid = summary["event_api_id"]
    people = summary["people"]

    if contacts:
        for p in people:
            enrich_contacts.enrich_person(p, use_web=contacts_web)
            enrich_contacts.save_person(eid, p)

    if rank:
        people = enrich.enrich_people(summary, people, use_web=rank_web)
        for p in people:
            if p.get("rank_score") is not None:
                save_ranking(eid, p["person_api_id"], p["rank_score"],
                             p["rank_reason"], p["icebreaker"])

    return summary, people
