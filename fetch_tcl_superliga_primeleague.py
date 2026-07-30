"""
fetch_tcl_superliga_primeleague.py
Queries the GRID Central Data GraphQL API to find all series for:
  - Misa               (TCL – Turkish Champions League)
  - UCAM Esports       (Superliga – Spain)
  - E WIE EINFACH E-SPORTS  (Prime League – Germany)
for games after 2026-03-01 (DATE_FROM), then downloads:

  Local:  end_state_summary_riot_{series_id}_{game_seq}.json
          end_state_details_riot_{series_id}_{game_seq}.json
  S3:     {S3_PREFIX}/events_{series_id}_{game_seq}_riot.jsonl

Workflow:
  Step 1 – scopri team/tournament IDs:
      python fetch_tcl_superliga_primeleague.py --discover

  Step 2 – incolla gli ID in TEAMS qui sotto, poi:
      python fetch_tcl_superliga_primeleague.py

  Oppure scarica solo un team:
      python fetch_tcl_superliga_primeleague.py --teams MISA
      python fetch_tcl_superliga_primeleague.py --teams UCAM EWE

Discovered IDs (via API introspection):
  ...da riempire dopo --discover...
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

# Solo partite a partire da questa data
DATE_FROM = "2026-06-7"

# ── Team config (riempire dopo --discover) ────────────────────────────────────
TEAMS: dict[str, dict] = {
    "MISA": {
        "label": "Misa (TCL – Turkey)",
        "team_id": "53133",          # es. "XXXXX"
        "tournament_ids": ['828727', '828822', '828833', '829240', '829587', '829647'], # es. ["XXXXX", "XXXXX"]
    },
    "UCAM": {
        "label": "UCAM Esports (Superliga – Spain)",
        "team_id": "52457",
        "tournament_ids": ['828825', '829106', '829108', '829647'],
    },
    "EWE": {
        "label": "E WIE EINFACH E-SPORTS (Prime League – Germany)",
        "team_id": "19899",
        "tournament_ids": ['829086', '829355', '829358', '829642', '829644', '829647'],
    },
}

# Keyword per trovare tornei e team in modalità --discover
DISCOVER_TOURNAMENT_KEYWORDS = ["tcl", "superliga", "prime league", "primeleague", "turkish"]
DISCOVER_TEAM_KEYWORDS = {
    "MISA": ["misa"],
    "UCAM": ["ucam"],
    "EWE":  ["einfach", "e wie"],
}

FORMAT_MAX_GAMES = {"Bo1": 1, "Bo2": 2, "Bo3": 3, "Bo5": 5, "Bo7": 7}

HEADERS = {
    "x-api-key": GRID_API_KEY,
    "Content-Type": "application/json",
}

# ── GraphQL queries ───────────────────────────────────────────────────────────
SERIES_BY_TEAM_QUERY = """
query FetchSeries($first: Int, $after: String, $teamId: ID, $tournamentIds: IdFilter) {
  allSeries(
    first: $first
    after: $after
    filter: {
      titleId: "3"
      types: ESPORTS
      teamId: $teamId
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

DISCOVERY_QUERY = """
query DiscoverSeries($first: Int, $after: String) {
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
        teams { baseInfo { id name } }
      }
    }
  }
}
"""


GQL_DELAY      = 1.6   # secondi tra ogni richiesta GQL (max 40/min = 1.5s)
GQL_RETRY_WAIT = 15.0
GQL_MAX_RETRIES = 4


def gql(query: str, variables: dict) -> dict:
    for attempt in range(GQL_MAX_RETRIES):
        resp = requests.post(
            GQL_URL,
            headers=HEADERS,
            json={"query": query, "variables": variables},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        if "errors" in body:
            # Rate limit dal layer GraphQL → aspetta e riprova
            err = body["errors"][0] if body["errors"] else {}
            if "rate" in err.get("message", "").lower() or "UNAVAILABLE" in str(err):
                wait = GQL_RETRY_WAIT * (attempt + 1)
                print(f"\n      GQL RATE-LIMITED – attendo {wait:.0f}s (tentativo {attempt + 1}/{GQL_MAX_RETRIES}) …")
                time.sleep(wait)
                continue
            raise RuntimeError(f"GraphQL errors: {json.dumps(body['errors'], indent=2)}")
        time.sleep(GQL_DELAY)
        return body["data"]
    raise RuntimeError(f"Ancora rate-limited GQL dopo {GQL_MAX_RETRIES} tentativi")


# ── Discovery ─────────────────────────────────────────────────────────────────
def discover() -> None:
    """
    Scansiona le series LoL dal 2026-03-01 in poi (DESC) e trova:
      - Tornei il cui nome contiene le keyword TCL/Superliga/Prime League
      - Team il cui nome corrisponde a Misa / UCAM / E WIE EINFACH
    Si ferma alle serie precedenti al 2026-03-01.
    """
    tourn_kw = [k.lower() for k in DISCOVER_TOURNAMENT_KEYWORDS]

    found_tournaments: dict[str, dict] = {}   # id → {name, date}
    found_teams: dict[str, dict[str, set]] = {
        k: {"ids": set(), "names": set(), "tournaments": set()}
        for k in DISCOVER_TEAM_KEYWORDS
    }

    after: str | None = None
    page = 0

    print(f"Scansione series LoL (DESC) — mi fermo prima del {DATE_FROM}\n")

    while True:
        page += 1
        variables = {"first": 50, "after": after}
        print(f"  Pagina {page}...", end=" ", flush=True)
        data = gql(DISCOVERY_QUERY, variables)
        conn = data["allSeries"]
        edges = conn["edges"]
        print(f"{len(edges)} series (totale={conn['totalCount']})")

        stop = False
        for edge in edges:
            node  = edge["node"]
            date  = node.get("startTimeScheduled", "")[:10]
            if date and date < DATE_FROM:
                stop = True
                break

            tourn = node.get("tournament") or {}
            tid   = tourn.get("id", "")
            tname = tourn.get("name", "")
            teams = node.get("teams") or []

            # Controlla se il torneo è rilevante
            tname_lower = tname.lower()
            if tid and any(kw in tname_lower for kw in tourn_kw):
                if tid not in found_tournaments:
                    found_tournaments[tid] = {"name": tname, "date": date}
                    # Cerca anche i team Misa/UCAM/EWE in questo torneo
                    for team_entry in teams:
                        bi = team_entry.get("baseInfo", {})
                        bname = bi.get("name", "")
                        bid   = bi.get("id", "")
                        bname_lower = bname.lower()
                        for key, kws in DISCOVER_TEAM_KEYWORDS.items():
                            if any(kw in bname_lower for kw in kws):
                                found_teams[key]["ids"].add(bid)
                                found_teams[key]["names"].add(bname)
                                found_teams[key]["tournaments"].add(tid)

            # Controlla anche i team indipendentemente dal torneo
            for team_entry in teams:
                bi = team_entry.get("baseInfo", {})
                bname = bi.get("name", "")
                bid   = bi.get("id", "")
                bname_lower = bname.lower()
                for key, kws in DISCOVER_TEAM_KEYWORDS.items():
                    if any(kw in bname_lower for kw in kws):
                        found_teams[key]["ids"].add(bid)
                        found_teams[key]["names"].add(bname)
                        if tid:
                            found_teams[key]["tournaments"].add(tid)
                        if tid and tid not in found_tournaments:
                            found_tournaments[tid] = {"name": tname, "date": date}

        if stop or not conn["pageInfo"]["hasNextPage"]:
            break
        after = conn["pageInfo"]["endCursor"]

    # ── Stampa risultati ──────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("TORNEI TCL / SUPERLIGA / PRIME LEAGUE trovati:\n")
    if found_tournaments:
        for tid, info in sorted(found_tournaments.items(), key=lambda x: x[1]["date"], reverse=True):
            print(f"  id={tid:<10}  {info['date']}  '{info['name']}'")
    else:
        print("  (nessuno — prova keyword diverse)")

    print(f"\n{'═'*65}")
    print("TEAM TARGET trovati:\n")
    for key, info in found_teams.items():
        team_label = TEAMS[key]["label"]
        print(f"  [{key}] {team_label}")
        if info["ids"]:
            for tid in sorted(info["ids"]):
                names = ", ".join(sorted(info["names"]))
                print(f"    team_id = \"{tid}\"   nome/i: {names}")
            print(f"    tournament_ids usati: {sorted(info['tournaments'])}")
        else:
            print("    (non trovato – controlla le keyword in DISCOVER_TEAM_KEYWORDS)")
        print()

    print(f"{'═'*65}")
    print("→ Copia team_id e tournament_ids in TEAMS nel file, poi lancia senza --discover.")


# ── Series fetching ───────────────────────────────────────────────────────────
def fetch_team_series(key: str, cfg: dict) -> list[dict]:
    """Restituisce le series del team con data >= DATE_FROM."""
    team_id      = cfg["team_id"]
    tourn_ids    = cfg["tournament_ids"]
    label        = cfg["label"]

    series: list[dict] = []
    after: str | None  = None
    page = 0

    variables_base: dict = {
        "first": 50,
        "teamId": team_id,
    }
    if tourn_ids:
        variables_base["tournamentIds"] = {"in": tourn_ids}

    while True:
        page += 1
        variables = {**variables_base, "after": after}
        print(f"  [{key}] Pagina {page}...", end=" ", flush=True)
        data = gql(SERIES_BY_TEAM_QUERY, variables)
        conn = data["allSeries"]
        edges = conn["edges"]
        print(f"{len(edges)} series (totale={conn['totalCount']})")

        for edge in edges:
            n    = edge["node"]
            date = n.get("startTimeScheduled", "")[:10]
            if date < DATE_FROM:
                continue
            teams = [t["baseInfo"]["name"] for t in (n.get("teams") or [])]
            fmt   = n.get("format", {}).get("nameShortened", "Bo1")
            tourn = n.get("tournament", {}).get("name", "")
            print(f"    id={n['id']}  {fmt}  {date}  '{tourn}'  {teams}")
            series.append(n)

        if not conn["pageInfo"]["hasNextPage"]:
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
        description="Scarica partite di Misa (TCL), UCAM (Superliga) ed E WIE EINFACH (Prime League)."
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Modalità discovery: trova team_id e tournament_ids da riempire in TEAMS.",
    )
    parser.add_argument(
        "--teams",
        nargs="+",
        default=list(TEAMS.keys()),
        choices=list(TEAMS.keys()),
        metavar="TEAM",
        help=f"Team da scaricare (default: tutti). Opzioni: {list(TEAMS.keys())}",
    )
    args = parser.parse_args()

    if args.discover:
        discover()
        return

    # Verifica che i team richiesti abbiano team_id configurato
    missing = [k for k in args.teams if not TEAMS[k]["team_id"]]
    if missing:
        print(f"ERRORE: team_id mancante per: {missing}")
        print("Lancia prima: python fetch_tcl_superliga_primeleague.py --discover")
        print("Poi inserisci team_id e tournament_ids in TEAMS nel file.")
        sys.exit(1)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name="eu-west-1",
    )

    total_series = 0
    for key in args.teams:
        cfg = TEAMS[key]
        print(f"\n{'═'*65}")
        print(f"=== {key}: {cfg['label']} (partite >= {DATE_FROM}) ===")
        print(f"{'═'*65}\n")

        print("── Step 1: query Central Data API ──")
        series_list = fetch_team_series(key, cfg)

        if not series_list:
            print(f"\nNessuna series trovata per {key} dopo {DATE_FROM}. Verifica gli ID.")
            continue

        print(f"\nTrovate {len(series_list)} series per {key}.\n")
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
                local_done = summary_path.exists() and details_path.exists()
                if local_done:
                    upload_events_to_s3(s3, events_url, s3_key)
                    continue
                sum_result = download_json(summary_url, summary_path)
                if sum_result is None and gn > 1:
                    print(f"      Game {gn} non esiste – series finita al game {gn - 1}")
                    break
                download_json(details_url, details_path)
                upload_events_to_s3(s3, events_url, s3_key)

    print(f"\n{'═'*65}")
    print(f"=== Completato – {total_series} series totali processate ===")


if __name__ == "__main__":
    main()
