"""
fetch_full_leagues.py
Scarica TUTTE le partite di LIT, LEC e LCK dalla GRID Central Data API.

Workflow in due passi:
  1. Scopri gli ID dei tornei:
         python fetch_full_leagues.py --discover

     Cerca i tornei LEC/LCK/LIT sull'API e stampa gli ID.
     Poi copia gli ID mancanti in LEAGUES qui sotto.

  2. Scarica tutto:
         python fetch_full_leagues.py

File salvati:
  Local:  end_state_summary_riot_{series_id}_{game_seq}.json
          end_state_details_riot_{series_id}_{game_seq}.json
  S3:     {S3_PREFIX}/events_{series_id}_{game_seq}_riot.jsonl

ID tornei già noti (da introspection precedente):
  LIT – LoL Italian Tournament - Winter 2026
    Regular Season → 827826
    Playoffs       → 827828
"""

import argparse
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
GRID_API_KEY = "9I7w6vKGjhw9TF0Vgr4oGwSbKUi7IEgvE4zYKRZC"
GQL_URL      = "https://api.grid.gg/central-data/graphql"
DL_BASE      = "https://api.grid.gg/file-download"

S3_BUCKET  = os.environ["S3_BUCKET"]
S3_PREFIX  = os.environ.get("S3_PREFIX", "Competitive")
AWS_KEY    = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]

# ── Definizione campionati ────────────────────────────────────────────────────
# Esegui prima `python fetch_full_leagues.py --discover` per trovare gli ID
# mancanti di LEC e LCK, poi riempili qui.
LEAGUES: dict[str, dict] = {
    "LIT": {
        "label": "LoL Italian Tournament - Winter 2026",
        "tournament_ids": [
            "827826",  # Regular Season
            "827828",  # Playoffs
        ],
    },
    "LEC": {
        "label": "LoL EMEA Championship - Spring 2026",
        "tournament_ids": [
            # TODO: inserire IDs dopo --discover
            # Esempio: "830001", "830002", ...
            "827701",
            "828729",
            "828973"
        ],
    },
    "LCK": {
        "label": "LoL Champions Korea - Spring 2026",
        "tournament_ids": [
            # TODO: inserire IDs dopo --discover
            "827844",
            "827846",
            "827848",
            "829039",
        ],
    },
}

FORMAT_MAX_GAMES = {"Bo1": 1, "Bo2": 2, "Bo3": 3, "Bo5": 5, "Bo7": 7}

HEADERS = {
    "x-api-key": GRID_API_KEY,
    "Content-Type": "application/json",
}

# ── GraphQL queries ───────────────────────────────────────────────────────────
SERIES_QUERY = """
query FetchAllSeries($first: Int, $after: String, $tournamentIds: IdFilter) {
  allSeries(
    first: $first
    after: $after
    filter: {
      titleId: "3"
      types: ESPORTS
      tournamentIds: $tournamentIds
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
"""

TOURNAMENT_DISCOVERY_QUERY = """
query DiscoverTournaments($first: Int, $after: String) {
  allSeries(
    first: $first
    after: $after
    filter: {
      titleId: "3"
      types: ESPORTS
    }
    orderBy: StartTimeScheduled
    orderDirection: DESC
  ) {
    totalCount
    pageInfo {
      hasNextPage
      endCursor
    }
    edges {
      node {
        startTimeScheduled
        tournament { id name }
      }
    }
  }
}
"""


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


# ── Discovery ─────────────────────────────────────────────────────────────────
def discover_tournaments(keywords: list[str]) -> None:
    """
    Scorre le series LoL del 2026 (ordinate DESC) e raccoglie i tornei unici
    il cui nome contiene almeno una delle keyword. Si ferma alla prima serie
    con data < 2026-01-01 per non scansionare anni precedenti.
    """
    keywords_lower = [k.lower() for k in keywords]
    print(f"Cercando tornei LoL 2026 con keyword: {keywords}\n")

    after: str | None = None
    page = 0
    seen_ids: set[str] = set()
    found: list[dict] = []

    while True:
        page += 1
        variables = {"first": 50, "after": after}
        print(f"  Pagina {page}...", end=" ", flush=True)
        data = gql(TOURNAMENT_DISCOVERY_QUERY, variables)
        conn = data["allSeries"]
        edges = conn["edges"]
        print(f"{len(edges)} series (totale={conn['totalCount']})")

        stop = False
        for edge in edges:
            node  = edge["node"]
            date  = node.get("startTimeScheduled", "")[:10]
            if date and date < "2026-01-01":
                stop = True
                break
            tourn = (node.get("tournament") or {})
            tid   = tourn.get("id")
            tname = tourn.get("name", "")
            if tid and tid not in seen_ids:
                seen_ids.add(tid)
                if any(kw in tname.lower() for kw in keywords_lower):
                    found.append({"id": tid, "name": tname, "date": date})

        if stop or not conn["pageInfo"]["hasNextPage"]:
            break
        after = conn["pageInfo"]["endCursor"]

    found.sort(key=lambda x: x["date"], reverse=True)
    print(f"\n{'─'*60}")
    if found:
        print(f"Trovati {len(found)} tornei 2026 corrispondenti alle keyword:\n")
        for t in found:
            print(f"  id={t['id']:<10}  {t['date']}  '{t['name']}'")
    else:
        print("Nessun torneo trovato. Prova keyword diverse.")
    print(f"\nCopia gli ID in LEAGUES nel file e poi lancia senza --discover.")


# ── Series fetching ───────────────────────────────────────────────────────────
def fetch_league_series(league_label: str, tournament_ids: list[str]) -> list[dict]:
    """
    Restituisce tutte le series già giocate (startTimeScheduled < oggi).
    Le series sono ordinate ASC, quindi ci fermiamo alla prima futura.
    """
    from datetime import date as _date
    today = _date.today().isoformat()  # es. "2026-04-22"

    series: list[dict] = []
    after: str | None = None
    page = 0

    while True:
        page += 1
        variables = {
            "first": 50,
            "after": after,
            "tournamentIds": {"in": tournament_ids},
        }
        print(f"  Pagina {page}...", end=" ", flush=True)
        data = gql(SERIES_QUERY, variables)
        conn = data["allSeries"]
        edges = conn["edges"]
        print(f"{len(edges)} series (totale={conn['totalCount']})")

        stop = False
        for edge in edges:
            n     = edge["node"]
            date  = n.get("startTimeScheduled", "")[:10]
            if date >= today:
                print(f"    STOP – series {n['id']} ({date}) è futura, interrompo")
                stop = True
                break
            teams = [t["baseInfo"]["name"] for t in (n.get("teams") or [])]
            fmt   = n.get("format", {}).get("nameShortened", "Bo1")
            tourn = n.get("tournament", {}).get("name", "")
            print(f"    id={n['id']}  {fmt}  {date}  '{tourn}'  {teams}")
            series.append(n)

        if stop or not conn["pageInfo"]["hasNextPage"]:
            break
        after = conn["pageInfo"]["endCursor"]

    return series


# ── Download helpers ──────────────────────────────────────────────────────────
DOWNLOAD_DELAY = 1.5
RETRY_WAIT     = 15.0
MAX_RETRIES    = 3


class NotFound(Exception):
    pass


def _get_with_retry(url: str, timeout: int = 60) -> requests.Response:
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers={"x-api-key": GRID_API_KEY}, timeout=timeout)
        if resp.status_code == 404:
            raise NotFound(url)
        if resp.status_code == 429:
            wait = RETRY_WAIT * (attempt + 1)
            print(f"      RATE-LIMITED – attendo {wait:.0f}s (tentativo {attempt + 1}/{MAX_RETRIES}) …")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(DOWNLOAD_DELAY)
        return resp
    raise RuntimeError(f"Ancora rate-limited dopo {MAX_RETRIES} tentativi: {url}")


def download_json(url: str, dest: Path) -> bool | None:
    if dest.exists():
        print(f"      SKIP (esiste) {dest.name}")
        return True
    try:
        resp = _get_with_retry(url)
        dest.write_bytes(resp.content)
        print(f"      SAVE {dest.name}  ({len(resp.content):,} bytes)")
        return True
    except NotFound:
        print(f"      404 (non trovato) {url}")
        return None
    except Exception as e:
        print(f"      ERRORE {url}: {e}")
        return False


def upload_events_to_s3(s3_client, url: str, s3_key: str) -> bool | None:
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        print(f"      SKIP (S3 esiste) {s3_key}")
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
        print(f"      404 events (non trovato) {url}")
        return None
    except Exception as e:
        print(f"      ERRORE events {url}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scarica tutte le partite di LIT, LEC e LCK dalla GRID API."
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Modalità discovery: elenca i tornei LoL e trova gli ID per LEC/LCK.",
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=list(LEAGUES.keys()),
        choices=list(LEAGUES.keys()),
        metavar="LEAGUE",
        help=f"Campionati da scaricare (default: tutti). Opzioni: {list(LEAGUES.keys())}",
    )
    args = parser.parse_args()

    if args.discover:
        discover_tournaments(["lec", "lck", "italian", "lit"])
        return

    # Verifica che ci siano IDs configurati per ogni lega richiesta
    missing = [lg for lg in args.leagues if not LEAGUES[lg]["tournament_ids"]]
    if missing:
        print(f"ERRORE: IDs tornei mancanti per: {missing}")
        print("Lancia prima: python fetch_full_leagues.py --discover")
        print("Poi inserisci gli ID in LEAGUES nel file.")
        sys.exit(1)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name="eu-west-1",
    )

    total_series = 0
    for league_key in args.leagues:
        league = LEAGUES[league_key]
        print(f"\n{'═'*60}")
        print(f"=== {league_key}: {league['label']} ===")
        print(f"{'═'*60}\n")

        print("── Step 1: query Central Data API ──")
        series_list = fetch_league_series(league["label"], league["tournament_ids"])

        if not series_list:
            print(f"\nNessuna series trovata per {league_key}. Verifica gli ID dei tornei.")
            continue

        print(f"\nTrovate {len(series_list)} series per {league_key}.\n")
        total_series += len(series_list)

        print("── Step 2: download file ──")
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
                if summary_path.exists() and details_path.exists():
                    print(f"      SKIP (già scaricato)")
                    continue
                sum_result = download_json(summary_url, summary_path)
                if sum_result is None and gn > 1:
                    print(f"      Game {gn} non esiste – series finita al game {gn - 1}")
                    break
                download_json(details_url, details_path)
                upload_events_to_s3(s3, events_url, s3_key)

    print(f"\n{'═'*60}")
    print(f"=== Completato – {total_series} series totali processate ===")


if __name__ == "__main__":
    main()
