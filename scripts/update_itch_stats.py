"""Fetch view/download counts for my itch.io games and save them for the website.

Runs in GitHub Actions (see .github/workflows/itch-stats.yml).
Needs the repository secret ITCH_API_KEY (itch.io → Settings → API keys).
Writes assets/data/itch-stats.json, which assets/site.js reads on the Hobby page.
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "data", "itch-stats.json")

key = os.environ.get("ITCH_API_KEY", "").strip()
if not key:
    sys.exit("ITCH_API_KEY is not set")

req = urllib.request.Request(
    f"https://itch.io/api/1/{key}/my-games",
    headers={"User-Agent": "rottenbutcher.github.io stats updater"},
)
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.load(r)

if "errors" in data:
    sys.exit(f"itch.io API error: {data['errors']}")

games = {}
for g in data.get("games", []):
    url = (g.get("url") or "").rstrip("/")
    games[url] = {
        "title": g.get("title"),
        "views": g.get("views_count", 0),
        "downloads": g.get("downloads_count", 0),
        "purchases": g.get("purchases_count", 0),
    }
    print(f"{g.get('title')}: views={g.get('views_count')} downloads={g.get('downloads_count')}")

# Only rewrite the file (and so only commit) when a number actually changed
old = {}
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        old = json.load(f).get("games", {})
if old == games:
    print("No change")
    sys.exit(0)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "games": games},
              f, ensure_ascii=False, indent=2)
print("Updated", OUT)
