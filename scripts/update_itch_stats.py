"""Fetch view/download counts for my itch.io games and save them for the website.

Runs in GitHub Actions (see .github/workflows/itch-stats.yml).
Needs the repository secret ITCH_API_KEY (itch.io → Settings → API keys).
Writes assets/data/itch-stats.json, which assets/site.js reads on the Hobby page.
"""
import html
import json
import re
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
        "cover": g.get("cover_url"),
        "short_text": g.get("short_text"),
        "type": g.get("type"),
        "published": g.get("published", False),
    }
    print(f"{g.get('title')}: views={g.get('views_count')} downloads={g.get('downloads_count')}")

# Read each game's public page for its description and genre (not in the API)
def page_info(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (rottenbutcher.github.io stats updater)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        page = r.read().decode("utf-8", "replace")
    info = {}
    m = re.search(r'<meta[^>]+property="og:description"[^>]+content="([^"]*)"', page) or \
        re.search(r'<meta[^>]+name="description"[^>]+content="([^"]*)"', page)
    if m:
        info["description"] = html.unescape(m.group(1)).strip()
    m = re.search(r'<td>\s*Genre\s*</td>\s*<td>(.*?)</td>', page, re.S)
    if m:
        info["genre"] = ", ".join(html.unescape(x).strip() for x in re.findall(r">([^<]+)</a>", m.group(1)))
    m = re.search(r'class="formatted_description[^"]*"[^>]*>(.*?)</div>', page, re.S)
    if m:
        text = re.sub(r"<[^>]+>", " ", m.group(1))
        info["long_description"] = re.sub(r"\s+", " ", html.unescape(text)).strip()[:1500]
    return info

for url, g in games.items():
    try:
        g.update(page_info(url))
    except Exception as e:
        print("page read failed for", url, e)

# Keep a local copy of each cover so the site does not depend on itch.io's image server
COVERS = os.path.join(os.path.dirname(OUT), "covers")
os.makedirs(COVERS, exist_ok=True)
for url, g in games.items():
    src = g.get("cover")
    if not src or not src.startswith("https://"):
        continue
    slug = url.rsplit("/", 1)[-1] or "game"
    ext = os.path.splitext(src.split("?")[0])[1].lower()
    ext = ext if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp") else ".png"
    dest = os.path.join(COVERS, slug + ext)
    if not os.path.exists(dest):
        try:
            creq = urllib.request.Request(src, headers={"User-Agent": "rottenbutcher.github.io stats updater"})
            with urllib.request.urlopen(creq, timeout=30) as r, open(dest, "wb") as f:
                f.write(r.read())
        except Exception as e:  # cover is optional
            print("cover download failed for", url, e)
            continue
    g["cover_local"] = "assets/data/covers/" + slug + ext

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
