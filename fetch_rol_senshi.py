"""
fetch_rol_senshi.py
Queries the GRID Central Data GraphQL API to find all Senshi eSports series in
Road of Legends - Summer 2026 (Benelux, League of Legends) scheduled from
2026-07-01 onwards, then downloads:

  Local:  end_state_summary_riot_{series_id}_{game_seq}.json
          end_state_details_riot_{series_id}_{game_seq}.json
  S3:     {S3_PREFIX}/events_{series_id}_{game_seq}_riot.jsonl

Discovered IDs (via API introspection):
  Team:        Senshi eSports → id=54486
  Tournaments: Road of Legends - Summer 2026
               Regular Season: Regular Season → id=829769
               Playoffs: Playoffs             → id=829771

Run:
    python fetch_rol_senshi.py
"""

import json
import os
import sys
import time
from pathlib import Path

import boto3
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent
GRID_API_KEY = "kD1D0BzNJxQLIapJkvDKNsq5UrSCGPVnNugdXLIS"
GQL_URL      = "https://api.grid.gg/central-data/graphql"
DL_BASE      = "https://api.grid.gg/file-download"

S3_BUCKET  = os.environ["S3_BUCKET"]
S3_PREFIX  = os.environ.get("S3_PREFIX", "Competitive")
AWS_KEY    = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]

# Discovered via API introspection
SENSHI_TEAM_ID = "54486"
TOURNAMENT_IDS = [
    # Road of Legends - Summer 2026 (Benelux)
    "829769",  # Regular Season: Regular Season
    "829771",  # Playoffs: Playoffs
    # "830960",  # EMEA Masters - Summer 2026 (Swiss: Swiss) – decommenta se serve
]
START_DATE = "2026-07-01T00:00:00Z"

# Maximum games to attempt per series format
FORMAT_MAX_GAMES = {"Bo1": 1, "Bo2": 2, "Bo3": 3, "Bo5": 5, "Bo7": 7}

HEADERS = {
    "x-api-key": GRID_API_KEY,
    "Content-Type": "application/json",
}

# ── GraphQL query ─────────────────────────────────────────────────────────────
SERIES_QUERY = """
query FetchSeries($first: Int, $after: String, $teamId: ID, $tournamentIds: IdFilter) {
  allSeries(
    first: $first
    after: $after
    filter: {
      titleId: "3"
      types: ESPORTS
      teamId: $teamId
      tournamentIds: $tournamentIds
      startTimeScheduled: { gte: "%s" }
    }
    orderBy: StartTimeScheduled
    orderDirection: ASC
  ) {
    totalCount
    pageInfo {
      hasNextPage
      endCursor
    }
    edges {
      node {
        id
        startTimeScheduled
        format { nameShortened }
        tournament { id name }
        teams { baseInfo { id name } }
      }
    }
  }
}
""" % START_DATE


def gql(query: str, variables: dict) -> dict:
    resp = requests.post(
        GQL_URL,
        headers=HEADERS,
        json={"query": query, "variables": variables},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    if "errors" in body:
        raise RuntimeError(f"GraphQL errors: {json.dumps(body['errors'], indent=2)}")
    return body["data"]


def fetch_senshi_rol_series() -> list[dict]:
    """Returns all Senshi eSports series in Road of Legends - Summer 2026 from START_DATE."""
    series: list[dict] = []
    after: str | None = None
    page = 0

    while True:
        page += 1
        variables = {
            "first": 50,
            "after": after,
            "teamId": SENSHI_TEAM_ID,
            "tournamentIds": {"in": TOURNAMENT_IDS},
        }
        print(f"  Querying page {page}...", end=" ", flush=True)
        data = gql(SERIES_QUERY, variables)
        conn = data["allSeries"]
        edges = conn["edges"]
        print(f"{len(edges)} series (total={conn['totalCount']})")

        for edge in edges:
            n = edge["node"]
            teams = [t["baseInfo"]["name"] for t in (n.get("teams") or [])]
            fmt   = n.get("format", {}).get("nameShortened", "Bo1")
            date  = n.get("startTimeScheduled", "")[:10]
            tourn = n.get("tournament", {}).get("name", "")
            print(f"    id={n['id']}  {fmt}  {date}  '{tourn}'  {teams}")
            series.append(n)

        if not conn["pageInfo"]["hasNextPage"]:
            break
        after = conn["pageInfo"]["endCursor"]

    return series


# ── File download helpers ─────────────────────────────────────────────────────
DOWNLOAD_DELAY = 1.5   # seconds between file-download requests to avoid 429s
RETRY_WAIT     = 15.0  # seconds to wait before retrying after a 429
MAX_RETRIES    = 3


class NotFound(Exception):
    """Raised when the server returns 404 (game truly does not exist)."""


def _get_with_retry(url: str, timeout: int = 60) -> requests.Response:
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers={"x-api-key": GRID_API_KEY}, timeout=timeout)
        if resp.status_code == 404:
            raise NotFound(url)
        if resp.status_code == 429:
            wait = RETRY_WAIT * (attempt + 1)
            print(f"      RATE-LIMITED – waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES}) …")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(DOWNLOAD_DELAY)
        return resp
    raise RuntimeError(f"Still rate-limited after {MAX_RETRIES} attempts: {url}")


def download_json(url: str, dest: Path) -> bool | None:
    if dest.exists():
        print(f"      SKIP (exists) {dest.name}")
        return True
    try:
        resp = _get_with_retry(url)
        dest.write_bytes(resp.content)
        print(f"      SAVE {dest.name}  ({len(resp.content):,} bytes)")
        return True
    except NotFound:
        print(f"      404 (not found) {url}")
        return None
    except Exception as e:
        print(f"      ERROR {url}: {e}")
        return False


def upload_events_to_s3(s3_client, url: str, s3_key: str) -> bool | None:
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        print(f"      SKIP (S3 exists) {s3_key}")
        return True
    except s3_client.exceptions.ClientError:
        pass

    try:
        resp = _get_with_retry(url, timeout=120)
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=resp.content,
            ContentType="application/x-ndjson",
        )
        print(f"      SAVE S3 {s3_key}  ({len(resp.content):,} bytes)")
        return True
    except NotFound:
        print(f"      404 events (not found) {url}")
        return None
    except Exception as e:
        print(f"      ERROR events {url}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== GRID – Senshi eSports / Road of Legends - Summer 2026 (from 2026-07-01) ===\n")

    print("── Step 1: querying Central Data API ──")
    series_list = fetch_senshi_rol_series()

    if not series_list:
        print("\nNo series found. Check team / tournament IDs.")
        sys.exit(0)

    print(f"\nFound {len(series_list)} series.\n")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name="eu-west-1",
    )

    print("── Step 2: downloading files ──")
    for series in series_list:
        sid    = series["id"]
        fmt    = (series.get("format") or {}).get("nameShortened", "Bo1")
        max_gn = FORMAT_MAX_GAMES.get(fmt, 1)
        teams  = [t["baseInfo"]["name"] for t in (series.get("teams") or [])]
        date   = series.get("startTimeScheduled", "")[:10]

        print(f"\n  series={sid}  {fmt} (max {max_gn} game(s))  {date}  {teams}")

        for gn in range(1, max_gn + 1):
            summary_url = f"{DL_BASE}/end-state/riot/series/{sid}/games/{gn}/summary"
            details_url = f"{DL_BASE}/end-state/riot/series/{sid}/games/{gn}/details"
            events_url  = f"{DL_BASE}/events/riot/series/{sid}/games/{gn}"

            summary_path = DATA_DIR / f"end_state_summary_riot_{sid}_{gn}.json"
            details_path = DATA_DIR / f"end_state_details_riot_{sid}_{gn}.json"
            s3_key       = f"{S3_PREFIX}/events_{sid}_{gn}_riot.jsonl"

            print(f"    game {gn}:")
            sum_result = download_json(summary_url, summary_path)
            if sum_result is None and gn > 1:
                print(f"      Game {gn} does not exist – series ended at game {gn - 1}")
                break
            download_json(details_url, details_path)
            upload_events_to_s3(s3, events_url, s3_key)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
